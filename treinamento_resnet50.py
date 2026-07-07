import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================== 
# FUNÇÃO PARA VISUALIZAR CLASSIFICAÇÕES
# ============================================================== 
def visualizar_predicoes(model, dataset, class_names, num_images=16, threshold=0.5):
    """
    Visualiza como o modelo está classificando as imagens
    
    Args:
        model: modelo treinado
        dataset: dataset para visualização
        class_names: nomes das classes ['non_drowsy', 'drowsy']
        num_images: número de imagens para visualizar
        threshold: limiar de decisão
    """
    # Coleta um batch de imagens
    for images, labels in dataset.take(1):
        # Faz predições
        predictions = model.predict(images)
        pred_classes = (predictions > threshold).astype(int).flatten()
        pred_probs = predictions.flatten()
        
        # Configura o grid de plotagem
        rows = int(np.ceil(num_images / 4))
        cols = min(4, num_images)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for i in range(min(num_images, len(images))):
            # Desnormaliza a imagem (assumindo normalização 0-1)
            img = images[i].numpy()
            
            # Mostra a imagem
            axes[i].imshow(img)
            
            # Define cores e texto baseado na predição
            true_label = int(labels[i])
            pred_label = pred_classes[i]
            prob = pred_probs[i]
            
            # Cor da borda: verde se acertou, vermelho se errou
            color = 'green' if true_label == pred_label else 'red'
            
            # Título com informações
            title = f"True: {class_names[true_label]}\n"
            title += f"Pred: {class_names[pred_label]} ({prob:.2f})"
            
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')
            
            # Adiciona barra de probabilidade
            prob_color = 'blue' if prob > 0.5 else 'orange'
            axes[i].text(0.5, -0.1, f'Prob: {prob:.2f}', 
                        transform=axes[i].transAxes, 
                        ha='center', color=prob_color, fontweight='bold', fontsize=10)

        # Esconde axes extras
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Visualização de Classificações (Threshold: {threshold})", fontsize=14, y=1.02)
        plt.tight_layout(pad=2)

        return fig

# ============================================================== 
# FUNÇÃO PARA ANALISAR INCERTEZAS
# ============================================================== 
def analise_de_predicoes_incertas(model, dataset, class_names, threshold_range=(0.3, 0.7)):
    """
    Analisa predições com baixa confiança (próximas ao threshold)
    """
    uncertain_images = []
    uncertain_probs = []
    uncertain_labels = []
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0).flatten()
        
        # Encontra predições incertas (probabilidades próximas a 0.5)
        for i, prob in enumerate(predictions):
            if threshold_range[0] <= prob <= threshold_range[1]:
                uncertain_images.append(images[i])
                uncertain_probs.append(prob)
                uncertain_labels.append(labels[i])
    
    if len(uncertain_images) > 0:
        # Mostra até 9 exemplos incertos
        num_display = min(9, len(uncertain_images))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_display):
            img = uncertain_images[i].numpy()
            prob = uncertain_probs[i]
            true_label = int(uncertain_labels[i])
            
            axes[i].imshow(img)
            axes[i].set_title(f"True: {class_names[true_label]}\nProb: {prob:.3f}", 
                            color='orange', fontsize=10)
            axes[i].axis('off')
        
        for i in range(num_display, 9):
            axes[i].axis('off')
        
        plt.suptitle(f"Predições Incertas ({len(uncertain_images)} encontradas)", fontsize=14)
        plt.tight_layout(pad=2)
        return fig
    
    return None


# ==============================================================
# TRAIN
# ==============================================================

def train():

    wandb.init(project="drowsiness_detection_resnet50")
    config = wandb.config

    DATASET_A = "../drowsiness_system/dataset"
    DATASET_B = "../drowsiness_system/nthuddd_dataset"
    CLASS_NAMES = ['non_drowsy', 'drowsy']  # Nomes das classes
    
    IMG_SIZE = (128,128)
    BATCH_SIZE = config.batch_size

    checkpoint_path = "melhor_modelo_resnet50.keras"

    # ==========================================================
    # DATASETS
    # ==========================================================

    def preprocess(image,label):
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        return image,label

    train_ds = image_dataset_from_directory(
        f"{DATASET_A}/train",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="int",
        shuffle=True
    ).map(preprocess)

    val_ds = image_dataset_from_directory(
        f"{DATASET_A}/val",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="int"
    ).map(preprocess)

    test_ds = image_dataset_from_directory(
        f"{DATASET_A}/test",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    ).map(preprocess)

    cross_val_ds = image_dataset_from_directory(
        f"{DATASET_B}/val",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    ).map(preprocess)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    cross_val_ds = cross_val_ds.prefetch(AUTOTUNE)

    # ==========================================================
    # MODEL
    # ==========================================================

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(128,128,3),
        pooling="avg"
    )

    for layer in base_model.layers[:-config.unfreeze_layers]:
        layer.trainable = False

    for layer in base_model.layers[-config.unfreeze_layers:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        
        Dense(config.dense_units,activation="relu",
              kernel_regularizer=regularizers.l2(config.l2)),
        
        Dropout(config.dropout),
        
        Dense(1,activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    callbacks = [
        EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
        ModelCheckpoint(checkpoint_path,monitor="val_loss",save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-6),
        TensorBoard(log_dir="logs_resnet50",histogram_freq=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks
    )

    # ==========================================================
    # Métricas e análise de overfitting
    # ==========================================================

    melhor_modelo = load_model(checkpoint_path)

    loss_treino = history.history["loss"][-1]
    acuracia_treino = history.history["accuracy"][-1]
    AUC_ROC_treino = history.history["auc"][-1]

    loss_validacao = history.history["val_loss"][-1]
    acuracia_validacao = history.history["val_accuracy"][-1]
    auc_validacao = history.history["val_auc"][-1]

    epocas = len(history.history["loss"])

    overfit_loss = loss_validacao - loss_treino
    overfit_accuracy = acuracia_treino - acuracia_validacao

    # ==========================================================
    # TESTE 
    # ==========================================================
    
    # ---------- YawnDD ----------
    y_verdadeiro = np.concatenate([y for _,y in test_ds])
    probabilidades = melhor_modelo.predict(test_ds).flatten()
    fpr, tpr, threshold = roc_curve(y_verdadeiro, probabilidades)
    melhor_thresh = threshold[np.argmax(tpr - fpr)]
    predicoes = (probabilidades>melhor_thresh).astype(int)

    precisao = precision_score(y_verdadeiro,predicoes)
    recall = recall_score(y_verdadeiro,predicoes)
    f1 = f1_score(y_verdadeiro,predicoes)
    auc_teste = roc_auc_score(y_verdadeiro,probabilidades)
    
    # ==============================================================
    # VISUALIZAÇÃO DAS CLASSIFICAÇÕES - YAWDD
    # ==============================================================
    
    # Visualização normal com threshold ótimo
    fig_pred = visualizar_predicoes(melhor_modelo, test_ds, CLASS_NAMES, 
                                     num_images=16, threshold=melhor_thresh)
    wandb.log({"predictions_yawdd": wandb.Image(fig_pred)})
    plt.close(fig_pred)
    
    # Análise de predições incertas
    fig_uncertain = analise_de_predicoes_incertas(melhor_modelo, test_ds, CLASS_NAMES)
    if fig_uncertain:
        wandb.log({"uncertain_predictions_yawdd": wandb.Image(fig_uncertain)})
        plt.close(fig_uncertain)
    
    # ==========================================================
    # Curva ROC yawnDD
    # ==========================================================

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_teste:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - YawnDD")
    plt.legend()

    wandb.log({"roc_curve_yawnDD": wandb.Image(plt)})
    plt.close()


    # ==========================================================
    # CROSS
    # ==========================================================
    
    # ---------- NTHUDD ----------
    y_verdadeiro_cross = np.concatenate([y for _,y in cross_val_ds])
    probabilidades_cross = melhor_modelo.predict(cross_val_ds).flatten()
    predicoes_cross = (probabilidades_cross>0.5).astype(int)

    auc_cross = roc_auc_score(y_verdadeiro_cross,probabilidades_cross)
    precision_score_cross = precision_score(y_verdadeiro_cross,predicoes_cross)
    recall_score_cross = recall_score(y_verdadeiro_cross,predicoes_cross)
    f1_score_cross = f1_score(y_verdadeiro_cross,predicoes_cross)
    
    
    balanced_acc = balanced_accuracy_score(y_verdadeiro,predicoes)
    balanced_acc_cross = balanced_accuracy_score(y_verdadeiro_cross,predicoes_cross)
    
    # ==============================================================
    # VISUALIZAÇÃO DAS CLASSIFICAÇÕES - NTHUDD
    # ==============================================================

    fig_pred_cross = visualizar_predicoes(melhor_modelo, cross_val_ds, CLASS_NAMES, 
                                num_images=16, threshold=0.5)
    wandb.log({"predictions_nthudd": wandb.Image(fig_pred_cross)})
    plt.close(fig_pred_cross)

    fig_uncertain_cross = analise_de_predicoes_incertas(melhor_modelo, cross_val_ds, CLASS_NAMES)

    if fig_uncertain_cross:
        wandb.log({"uncertain_predictions_nthudd": wandb.Image(fig_uncertain_cross)})
        plt.close(fig_uncertain_cross)

    # ==========================================================
    # Matriz de confusão
    # ==========================================================

    cm = confusion_matrix(y_verdadeiro,predicoes)
    plt.figure()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title("Confusion Matriz YawnDD")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    wandb.log({"matriz_confusao_yawndd":wandb.Image(plt)})
    plt.close()

    cm = confusion_matrix(y_verdadeiro_cross,predicoes_cross)
    plt.figure()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title("Confusion Matriz NTHUDD")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    wandb.log({"matriz_confusao_nthudd":wandb.Image(plt)})
    plt.close()
    
    

    # ==========================================================
    # WANDB LOG
    # ==========================================================

    wandb.log({
        "balanced_accuracy": balanced_acc,
        "balanced_accuracy_cross": balanced_acc_cross,

        "best_threshold": float(melhor_thresh),
        "train_loss":loss_treino,
        "train_accuracy":acuracia_treino,
        "train_auc":AUC_ROC_treino,

        "val_loss":loss_validacao,
        "val_accuracy":acuracia_validacao,
        "val_auc":auc_validacao,

        "epochs":epocas,

        "overfit_loss":overfit_loss,
        "overfit_accuracy":overfit_accuracy,

        "precision_database":precisao,
        "recall_database":recall,
        "f1_database":f1,

        "precision_cross":precision_score_cross,
        "recall_cross":recall_score_cross,
        "f1_cross":f1_score_cross,

        "test_auc":auc_teste,
        "cross_auc":auc_cross
    })

    # ==========================================================
    # PROMOTION
    # ==========================================================

    if loss_validacao < 0.3 and overfit_loss < 0.1:

        melhor_modelo.save("melhor_modelo_resnet50.keras")

        art = wandb.Artifact(
            "melhor_modelo_resnet50",
            type="model",
            metadata={
                "val_loss":loss_validacao,
                "val_auc":auc_validacao,
                "overfit":overfit_loss
            }
        )

        art.add_file("melhor_modelo_resnet50.keras")
        wandb.log_artifact(art)

        print("modelo promovido para o repositório de modelos")

    else:
        print("modelo descartado")

    wandb.finish()

# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    train()
