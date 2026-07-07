
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential,load_model
from keras import regularizers, mixed_precision
from keras.optimizers import AdamW, Adam, RMSprop, SGD, Adadelta
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, balanced_accuracy_score
import wandb
from wandb.integration.keras import WandbMetricsLogger
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc


def denormalizar_efficientnet(img):
    img = np.array(img, dtype=np.float32).copy()
    
    if img.max() > 1.5:
        img /= 255.0
    
    if img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    else:
        img = np.clip(img, 0.0, 1.0)
    
    return img

def analise_distribuicao_probabilidades(model, dataset, class_names, bins=50, figsize=(12, 5)):
    """
    Analisa a distribuição das probabilidades preditas e ajuda a escolher o melhor threshold.
    
    Args:
        model: modelo treinado
        dataset: dataset (com labels)
        class_names: nomes das classes
        bins: número de bins para o histograma
        figsize: tamanho da figura
    
    Returns:
        dict com estatísticas e figura
    """
    # Coleta todas as probabilidades e labels
    y_true = []
    y_probs = []
    
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).flatten()
        y_probs.extend(probs)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Separa probabilidades por classe
    probs_classe_0 = y_probs[y_true == 0]  # non_drowsy
    probs_classe_1 = y_probs[y_true == 1]  # drowsy
    
    # Calcula métricas para diferentes thresholds
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        preds = (y_probs > thresh).astype(int)
        accuracies.append(accuracy_score(y_true, preds))
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1_scores.append(f1_score(y_true, preds, zero_division=0))
    
    # Encontra melhor threshold por F1-score
    best_idx = np.argmax(f1_scores)
    best_thresh_f1 = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Encontra threshold que balanceia precisão e recall (Youden's J)
    youden_idx = np.argmax(np.array(recalls) + np.array(precisions) - 1)
    best_thresh_youden = thresholds[youden_idx]
    
    # Calcula estatísticas
    stats = {
        'threshold_f1': best_thresh_f1,
        'threshold_youden': best_thresh_youden,
        'best_f1': best_f1,
        'mean_prob_class_0': np.mean(probs_classe_0),
        'std_prob_class_0': np.std(probs_classe_0),
        'mean_prob_class_1': np.mean(probs_classe_1),
        'std_prob_class_1': np.std(probs_classe_1),
        'overlap_area': calcular_sobreposicao(probs_classe_0, probs_classe_1)
    }
    
    # Cria visualização
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histograma das probabilidades por classe
    axes[0].hist(probs_classe_0, bins=bins, alpha=0.7, label=class_names[0], color='blue', density=True)
    axes[0].hist(probs_classe_1, bins=bins, alpha=0.7, label=class_names[1], color='red', density=True)
    axes[0].axvline(x=best_thresh_f1, color='green', linestyle='--', linewidth=2, label=f'Melhor F1-score (thresh={best_thresh_f1:.3f})')
    axes[0].axvline(x=best_thresh_youden, color='orange', linestyle=':', linewidth=2, label=f'Youden (thresh={best_thresh_youden:.3f})')
    axes[0].set_xlabel('Probabilidade (classe positiva: sonolento)')
    axes[0].set_ylabel('Densidade')
    axes[0].set_title('Distribuição de Probabilidades por Classe')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Curvas de métricas vs threshold
    axes[1].plot(thresholds, accuracies, label='Acurácia', linewidth=2)
    axes[1].plot(thresholds, precisions, label='Precisão', linewidth=2)
    axes[1].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[1].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    axes[1].axvline(x=best_thresh_f1, color='green', linestyle='--', linewidth=2, label=f'Melhor F1-score ({best_thresh_f1:.3f})')
    axes[1].axvline(x=best_thresh_youden, color='orange', linestyle=':', linewidth=2, label=f'Youden ({best_thresh_youden:.3f})')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Métrica')
    axes[1].set_title('Métricas vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    
    # Reserva espaço inferior para exibir estatísticas sem sobrepor os subplots
    plt.tight_layout(rect=[0, 0.26, 1, 0.98])
    

    # Adiciona texto com estatísticas
    stats_text = (f"Estatísticas:\n"
                  f"Classe '{class_names[0]}': μ={stats['mean_prob_class_0']:.3f}, σ={stats['std_prob_class_0']:.3f}\n"
                  f"Classe '{class_names[1]}': μ={stats['mean_prob_class_1']:.3f}, σ={stats['std_prob_class_1']:.3f}\n"
                  f"Sobreposição: {stats['overlap_area']:.3f}\n"
                  f"Threshold recomendado (F1): {best_thresh_f1:.3f}\n"
                  f"Threshold recomendado (Youden): {best_thresh_youden:.3f}")
    
    fig.text(
        0.5, 0.02, stats_text,
        ha='center', va='bottom', fontsize=9, linespacing=1.3,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    return stats, fig

def calcular_sobreposicao(probs_a, probs_b, bins=100):
    """
    Calcula a área de sobreposição entre duas distribuições.
    """
    # Cria histogramas
    hist_a, bin_edges = np.histogram(probs_a, bins=bins, density=True)
    hist_b, _ = np.histogram(probs_b, bins=bin_edges, density=True)
    
    # Área de sobreposição
    overlap = np.sum(np.minimum(hist_a, hist_b) * np.diff(bin_edges))
    return overlap

def avaliar_threshold_com_curva_risco(model, dataset, class_names, custo_fp=1.0, custo_fn=1.0):
    """
    Avalia threshold considerando custos de falsos positivos e falsos negativos.
    
    Args:
        model: modelo treinado
        dataset: dataset com labels
        class_names: nomes das classes
        custo_fp: custo do falso positivo (ex: custo de alarme falso)
        custo_fn: custo do falso negativo (ex: custo de não detectar sonolência)
    
    Returns:
        threshold ótimo que minimiza o custo total
    """
    # Coleta predições
    y_true = []
    y_probs = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).flatten()
        y_probs.extend(probs)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Calcula matriz de confusão para cada threshold
    thresholds = np.linspace(0, 1, 200)
    custos = []
    
    for thresh in thresholds:
        preds = (y_probs > thresh).astype(int)
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tp = np.sum((preds == 1) & (y_true == 1))
        
        # Custo total: fp * custo_fp + fn * custo_fn
        custo_total = fp * custo_fp + fn * custo_fn
        custos.append(custo_total)
    
    best_idx = np.argmin(custos)
    best_thresh = thresholds[best_idx]
    min_custo = custos[best_idx]
    
    # Visualização
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, custos, linewidth=2, label='Custo Total')
    ax.axvline(x=best_thresh, color='red', linestyle='--', 
               label=f'Threshold ótimo ({best_thresh:.3f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Custo Total')
    ax.set_title(f'Análise de Custo (FP={custo_fp}, FN={custo_fn})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adiciona informações
    info_text = (f"Melhor threshold: {best_thresh:.3f}\n"
                 f"Custo mínimo: {min_custo:.2f}\n"
                 f"Relação custo FP/FN: {custo_fp/custo_fn:.2f}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return best_thresh, min_custo, fig

# ==============================================================
# FUNÇÕES DE VISUALIZAÇÃO E ANÁLISE
# ==============================================================

def visualizar_predicoes(model, dataset, class_names, num_images=16, threshold=0.5):
    """Visualiza predições do modelo."""
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        pred_classes = (predictions > threshold).astype(int).flatten()
        pred_probs = predictions.flatten()

        rows = int(np.ceil(num_images / 4))
        cols = min(4, num_images)
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axes = axes.flatten() if num_images > 1 else [axes]

        for i in range(min(num_images, len(images))):
            img = images[i].numpy().copy()

            # desfaz preprocess_input do ResNet50
            img = denormalizar_efficientnet(img)

            axes[i].imshow(img)

            true_label = int(labels[i])
            pred_label = pred_classes[i]
            prob = pred_probs[i]
            color = 'green' if true_label == pred_label else 'red'
            title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({prob:.2f})"
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')

            prob_color = 'blue' if prob > 0.5 else 'orange'
            axes[i].text(0.5, -0.1, f'Prob: {prob:.2f}',
                        transform=axes[i].transAxes,
                        ha='center', color=prob_color, fontweight='bold', fontsize=10)

        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Visualização de Classificações (Threshold: {threshold})", fontsize=14, y=1.02)
        plt.tight_layout(pad=2)
        return fig

def analise_de_predicoes_incertas(model, dataset, class_names, threshold_range=(0.3, 0.7)):
    """Analisa predições com baixa confiança."""
    uncertain_images, uncertain_probs, uncertain_labels = [], [], []
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0).flatten()
        for i, prob in enumerate(predictions):
            if threshold_range[0] <= prob <= threshold_range[1]:
                uncertain_images.append(images[i])
                uncertain_probs.append(prob)
                uncertain_labels.append(labels[i])

    if len(uncertain_images) > 0:
        num_display = min(9, len(uncertain_images))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(num_display):
            img = uncertain_images[i].numpy().copy()
            img = denormalizar_efficientnet(img)

            prob = uncertain_probs[i]
            true_label = int(uncertain_labels[i])

            axes[i].imshow(img)
            axes[i].set_title(
                f"True: {class_names[true_label]}\nProb: {prob:.3f}",
                color='orange',
                fontsize=10
            )
            axes[i].axis('off')

        for i in range(num_display, 9):
            axes[i].axis('off')

        plt.suptitle(f"Predições Incertas ({len(uncertain_images)} encontradas)", fontsize=14)
        plt.tight_layout(pad=2)
        return fig

    return None

# ==============================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ==============================================================

def train():
    wandb.init(project="efficientnet")
    config = wandb.config

    DATASET_A = "../drowsiness_system/dataset_personalizado_v2"
    DATASET_B = "../drowsiness_system/nthuddd_dataset"
    IMG_SIZE = (config.img_size, config.img_size)
    BATCH_SIZE = config.batch_size
    CLASS_NAMES = ['non_drowsy', 'drowsy']
    checkpoint_path = "melhor_modelo_efficientnet.keras"

    # ========== CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ==========
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        if image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        return image, label

    # ========== DATA AUGMENTATION ==========
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ])

    train_ds = image_dataset_from_directory(
        f"{DATASET_A}/train", labels="inferred", label_mode="int",
        batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True, color_mode="rgb", seed=42
    )
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = image_dataset_from_directory(
        f"{DATASET_A}/val", labels="inferred", label_mode="int",
        batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True, color_mode="rgb", seed=42
    ).map(preprocess)

    cross_val_ds = image_dataset_from_directory(
        f"{DATASET_B}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,color_mode="rgb"
    ).map(preprocess)

    test_ds = image_dataset_from_directory(
        f"{DATASET_A}/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        color_mode="rgb", shuffle=False
    ).map(preprocess)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    cross_val_ds = cross_val_ds.prefetch(AUTOTUNE)

    # ========== CONSTRUÇÃO DO MODELO ==========
    # strategy = tf.distribute.MirroredStrategy() -> para treinamento em múltiplas GPUs (opcional)
    #  with strategy.scope():

    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = True
    for layer in base_model.layers[:-config.unfreeze_layers]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(config.dense_units, activation="relu",
                kernel_regularizer=regularizers.l2(config.l2)),
        Dropout(config.dropout),
        Dense(1, activation="sigmoid",dtype='float32') 
    ])

    optimizer = AdamW(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss= tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.PrecisionAtRecall(name="precision_at_recall_for_80", recall=0.8),
            tf.keras.metrics.SpecificityAtSensitivity(name="specificity_at_sensitivity_for_80", sensitivity=0.8)
        ]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True,
                    write_images=False, write_steps_per_second=True,
                    update_freq="epoch", profile_batch=0),
        WandbMetricsLogger(log_freq="epoch")
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks
    )

    # ========== AVALIAÇÃO E MÉTRICAS ==========
    melhor_modelo = load_model(checkpoint_path)
    results_cross = melhor_modelo.evaluate(cross_val_ds, verbose=0)
    results_test = melhor_modelo.evaluate(test_ds, verbose=0)

    def ultimo_valor(history_dict, *keys):
        for key in keys:
            if key in history_dict:
                return history_dict[key][-1]
        raise KeyError(f"Nenhuma das chaves {keys} encontrada. Disponíveis: {list(history_dict.keys())}")

    loss_treino = ultimo_valor(history.history, "loss")
    acuracia_treino = ultimo_valor(history.history, "accuracy")
    auc_treino = ultimo_valor(history.history, "auc")
    loss_val = ultimo_valor(history.history, "val_loss")
    acuracia_val = ultimo_valor(history.history, "val_accuracy")
    auc_val = ultimo_valor(history.history, "val_auc")
    epocas = len(history.history["loss"])
    overfit_loss = loss_val - loss_treino
    overfit_accuracy = acuracia_treino - acuracia_val
    test_loss = results_test[0]
    test_acc = results_test[1]
    test_precision = results_test[3]
    test_recall = results_test[4]
    test_f1 = 2*(test_precision * test_recall) / (test_precision + test_recall + 1e-10)


    # ========== PREDIÇÕES NO DATASET DE TESTE (YawDD) ==========
    y_true = np.concatenate([y for _, y in test_ds])
    probs = melhor_modelo.predict(test_ds).flatten()
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    melhor_thresh = thresholds[np.argmax(tpr - fpr)]
    preds = (probs > melhor_thresh).astype(int)

    precisao = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    auc_teste = roc_auc_score(y_true, probs)
    balanced_acc = balanced_accuracy_score(y_true, preds)

    # ========== PREDIÇÕES NO DATASET CRUZADO (NTHUDD) ==========
    y_true_cross = np.concatenate([y for _, y in cross_val_ds])
    probs_cross = melhor_modelo.predict(cross_val_ds).flatten()
    fpr_cross, tpr_cross, thresholds_cross = roc_curve(y_true_cross, probs_cross)
    preds_cross = (probs_cross > melhor_thresh).astype(int)

    auc_cross = roc_auc_score(y_true_cross, probs_cross)
    precisao_cross = precision_score(y_true_cross, preds_cross)
    recall_cross = recall_score(y_true_cross, preds_cross)
    f1_cross = f1_score(y_true_cross, preds_cross)
    balanced_acc_cross = balanced_accuracy_score(y_true_cross, preds_cross)

    # ========== VISUALIZAÇÕES E LOG NO WANDB ==========
    # Visualizações YawDD
    fig_pred = visualizar_predicoes(melhor_modelo, test_ds, CLASS_NAMES,
                                    num_images=16, threshold=melhor_thresh)
    wandb.log({"predictions_yawdd": wandb.Image(fig_pred)})
    plt.close(fig_pred)

    fig_uncertain = analise_de_predicoes_incertas(melhor_modelo, test_ds, CLASS_NAMES)
    if fig_uncertain:
        wandb.log({"uncertain_predictions_yawdd": wandb.Image(fig_uncertain)})
        plt.close(fig_uncertain)
        
     # ========== ANÁLISE DE DISTRIBUIÇÃO DE PROBABILIDADES ==========
    # Para YawDD
    stats_yawdd, fig_dist_yawdd = analise_distribuicao_probabilidades(
        melhor_modelo, test_ds, CLASS_NAMES
    )
    wandb.log({"distribuicao_probabilidades_yawdd": wandb.Image(fig_dist_yawdd)})
    plt.close(fig_dist_yawdd)

    # Para NTHUDD
    stats_nthudd, fig_dist_nthudd = analise_distribuicao_probabilidades(
        melhor_modelo, cross_val_ds, CLASS_NAMES
    )
    wandb.log({"distribuicao_probabilidades_nthudd": wandb.Image(fig_dist_nthudd)})
    plt.close(fig_dist_nthudd)

    # Análise de custo (opcional - ajuste os custos conforme sua aplicação)
    # Exemplo: custo de falso negativo (não detectar sonolência) é maior que falso positivo
    custo_fp = 1.0   # custo de alarme falso
    custo_fn = 5.0   # custo de não detectar sonolência
    
    threshold_custo, min_custo, fig_custo = avaliar_threshold_com_curva_risco(
        melhor_modelo, test_ds, CLASS_NAMES, custo_fp, custo_fn
    )
    wandb.log({"analise_custo_threshold": wandb.Image(fig_custo)})
    plt.close(fig_custo)

    print(f"\nThresholds recomendados:")
    print(f"  - Por F1-score: {stats_yawdd['threshold_f1']:.3f}")
    print(f"  - Por Youden's J: {stats_yawdd['threshold_youden']:.3f}")
    print(f"  - Por custo (FP={custo_fp}, FN={custo_fn}): {threshold_custo:.3f}")

    # Curva ROC YawDD
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_teste:.3f}")
    plt.plot([0,1], [0,1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - YawDD")
    plt.legend()
    wandb.log({"roc_curve_yawDD": wandb.Image(plt)})
    plt.close()
    
    # Curva ROC NTHUDD
    plt.figure()
    plt.plot(fpr_cross, tpr_cross, label=f"AUC={auc_cross:.3f}")
    plt.plot([0,1], [0,1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - NTHUDDD")
    plt.legend()
    wandb.log({"roc_curve_nthuddd": wandb.Image(plt)})
    plt.close()
    

    # Visualizações NTHUDD
    fig_pred_cross = visualizar_predicoes(melhor_modelo, cross_val_ds, CLASS_NAMES,
                                          num_images=16, threshold=melhor_thresh)
    wandb.log({"predictions_nthuddd": wandb.Image(fig_pred_cross)})
    plt.close(fig_pred_cross)

    fig_uncertain_cross = analise_de_predicoes_incertas(melhor_modelo, cross_val_ds, CLASS_NAMES)
    if fig_uncertain_cross:
        wandb.log({"uncertain_predictions_nthuddd": wandb.Image(fig_uncertain_cross)})
        plt.close(fig_uncertain_cross)

    # Matrizes de confusão
    cm = confusion_matrix(y_true, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix YawDD")
    plt.xlabel("Predict")
    plt.ylabel("Real")
    wandb.log({"matriz_confusao_yawdd": wandb.Image(plt)})
    plt.close()

    cm_cross = confusion_matrix(y_true_cross, preds_cross)
    plt.figure()
    sns.heatmap(cm_cross, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix NTHUDDD")
    plt.xlabel("Predict")
    plt.ylabel("Real")
    wandb.log({"matriz_confusao_nthuddd": wandb.Image(plt)})
    plt.close()


    # ========== LOG DE MÉTRICAS ==========
    wandb.log({
        "balanced_accuracy": balanced_acc,
        "balanced_accuracy_cross": balanced_acc_cross,
        "best_threshold": float(melhor_thresh),
        "train_loss": loss_treino,
        "train_accuracy": acuracia_treino,
        "train_auc": auc_treino,
        "val_loss": loss_val,
        "val_accuracy": acuracia_val,
        "val_auc": auc_val,
        "epochs": epocas,
        "overfit_loss": overfit_loss,
        "overfit_accuracy": overfit_accuracy,
        "precision_database": precisao,
        "recall_database": recall,
        "f1_database": f1,
        "precision_cross": precisao_cross,
        "recall_cross": recall_cross,
        "f1_cross": f1_cross,
        "test_auc": auc_teste,
        'test_precision': test_precision,
        'test_recall': test_recall,
        "test_f1": test_f1,
        "test_accuracy": test_acc,
        "cross_auc": auc_cross,
        "False Positives": ultimo_valor(history.history, "fp"),
        "False Negatives": ultimo_valor(history.history, "fn"),
        "True Positives": ultimo_valor(history.history, "tp"),
        "True Negatives": ultimo_valor(history.history, "tn"),
        "precision_at_recall_80": ultimo_valor(history.history, "precision_at_recall_for_80"),
        "specificity_at_sensitivity_80": ultimo_valor(history.history, "specificity_at_sensitivity_for_80")
    })

    # ========= PROMOÇÃO DO MODELO (SE APROVADO) ==========
    melhor_modelo.save("melhor_modelo_efficientnetb0.keras")
    art = wandb.Artifact(
        "melhor_modelo_efficientnetb0",
        type="model",
        metadata={"val_loss": loss_val, "val_auc": auc_val, "overfit": overfit_loss}
    )
    with open("melhor_threshold.json", "w") as f:
        json.dump({"threshold": float(melhor_thresh)}, f)
    art.add_file("melhor_threshold.json") 
    
    art.add_file("melhor_modelo_efficientnetb0.keras")
    wandb.log_artifact(art)

    wandb.finish()

    # ========== RESULTADOS FINAIS ==========
    print("\n========================================================")
    print(" Avaliação na base YawDD:")
    print(f"Acurácia: {test_acc:.4f} | Precisão: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    print("========================================================")
    print("\n========================================================")
    print(" Avaliação na base NTHUDD:")
    print(f"Acurácia: {results_cross[1]:.4f} | Precisão: {precisao_cross:.4f} | Recall: {recall_cross:.4f} | F1: {f1_cross:.4f}")
    print("========================================================")

# ==============================================================
# EXECUÇÃO
# ==============================================================
if __name__ == "__main__":
    train()