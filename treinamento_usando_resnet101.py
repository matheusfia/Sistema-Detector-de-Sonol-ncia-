import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import regularizers, mixed_precision
from keras.optimizers import AdamW
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.applications import ResNet101
from keras.applications.resnet import preprocess_input
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, balanced_accuracy_score
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import gc

import modelo

# Configuração de GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.backend.clear_session()
gc.collect()

# Configuração de mixed precision para acelerar o treinamento em GPUs compatíveis
mixed_precision.set_global_policy('mixed_float16')

def denormalizar_resnet_vgg(img):
    img = img.copy()
    
    # desfaz mean subtraction (BGR)
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 123.68

    # BGR → RGB
    img = img[..., ::-1]

    # normaliza pra [0,1]
    img = np.clip(img / 255.0, 0, 1)

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
# FUNÇÕES AUXILIARES PARA GRAD-CAM
# ==============================================================

def sobrepor_mapa_calor(imagem, mapa_calor, alpha=0.4):
    """Sobrepõe o mapa de calor na imagem original."""
    mapa_calor = cv2.resize(mapa_calor, (imagem.shape[1], imagem.shape[0]))
    mapa_calor = np.uint8(255 * mapa_calor)
    mapa_calor = cv2.applyColorMap(mapa_calor, cv2.COLORMAP_JET)
    if imagem.dtype != np.uint8:
        if imagem.max() <= 1.0:
            imagem = np.uint8(255 * imagem)
        else:
            imagem = np.uint8(imagem)
    return cv2.addWeighted(imagem, 1 - alpha, mapa_calor, alpha, 0)

def logar_gradcam_final(modelo, dataset, class_names, num_imagens=4):
    """Gera e loga mapas Grad-CAM para o modelo binário (ResNet101 + sigmoid)."""
    try:
        # =========================================================
        # 1) FORÇA O MODELO A SER CHAMADO PARA DEFINIR input/output
        # =========================================================
        for imagens, _ in dataset.take(1):
            _ = modelo(imagens[:1], training=False)
            break

        # =========================================================
        # 2) PEGA O BACKBONE RESNET50 DENTRO DO SEQUENTIAL
        # =========================================================
        base_model = None
        for layer in modelo.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            raise ValueError("ResNet101 não encontrado no modelo carregado")

        print(f"Base model encontrado: {base_model.name}")

        # =========================================================
        # 3) PEGA A ÚLTIMA CAMADA CONVOLUCIONAL
        # =========================================================
        ultima_camada = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                ultima_camada = layer
                break

        if ultima_camada is None:
            raise ValueError("Nenhuma camada convolucional encontrada no ResNet101")

        print(f"Última camada convolucional: {ultima_camada.name}")
        # =========================================================
        # 4) CRIA MODELO AUXILIAR DE GRAD-CAM
        # =========================================================
        grad_model = tf.keras.models.Model(
            inputs=modelo.input,
            outputs=[base_model.get_layer(ultima_camada.name).output, modelo.output]
)
        # =========================================================
        # 5) GERA O GRAD-CAM
        # =========================================================
        for imagens, rotulos in dataset.take(1):
            for i in range(min(num_imagens, len(imagens))):
                img = tf.expand_dims(imagens[i], axis=0)

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img, training=False)
                    loss = predictions[:, 0]  # probabilidade da classe positiva

                grads = tape.gradient(loss, conv_outputs)
                if grads is None:
                    print(f"Gradientes nulos para imagem {i}")
                    continue

                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]

                heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
                heatmap = tf.maximum(heatmap, 0)
                heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
                heatmap = heatmap.numpy()

                img_visual = imagens[i].numpy().copy()
                img_visual = denormalizar_resnet_vgg(img_visual)

                imagem_final = sobrepor_mapa_calor(img_visual, heatmap)

                true_label = class_names[int(rotulos[i])]
                pred_prob = float(predictions[0][0])

                wandb.log({
                    f"gradcam_{true_label}_exemplo_{i}": wandb.Image(
                        imagem_final,
                        caption=f"True: {true_label} | Prob drowsy: {pred_prob:.3f}"
                    )
                })

                print(f"✓ Grad-CAM gerado para imagem {i} (label: {true_label})")
            break

    except Exception as e:
        print(f"Erro no Grad-CAM: {e}")

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
            img = denormalizar_resnet_vgg(img)

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
            img = denormalizar_resnet_vgg(img)

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
    wandb.init(project="resnet101")
    config = wandb.config

    DATASET_A = "../drowsiness_system/dataset_personalizado_v2"
    DATASET_B = "../drowsiness_system/nthuddd_dataset"
    IMG_SIZE = (config.img_size, config.img_size)
    BATCH_SIZE = config.batch_size
    CLASS_NAMES = ['non_drowsy', 'drowsy']
    checkpoint_path = "melhor_modelo_resnet101.keras"

    # ========== CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ==========
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
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
        f"{DATASET_B}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        color_mode="rgb", shuffle=False
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

    base_model = ResNet101(
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
        loss="binary_crossentropy",
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
                    update_freq="epoch", profile_batch=0)
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
    plt.title("ROC - YawnDD")
    plt.legend()
    wandb.log({"roc_curve_yawnDD": wandb.Image(plt)})
    plt.close()

    # Visualizações NTHUDD
    fig_pred_cross = visualizar_predicoes(melhor_modelo, cross_val_ds, CLASS_NAMES,
                                          num_images=16, threshold=melhor_thresh)
    wandb.log({"predictions_nthudd": wandb.Image(fig_pred_cross)})
    plt.close(fig_pred_cross)

    fig_uncertain_cross = analise_de_predicoes_incertas(melhor_modelo, cross_val_ds, CLASS_NAMES)
    if fig_uncertain_cross:
        wandb.log({"uncertain_predictions_nthudd": wandb.Image(fig_uncertain_cross)})
        plt.close(fig_uncertain_cross)

    # Matrizes de confusão
    cm = confusion_matrix(y_true, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix YawnDD")
    plt.xlabel("Predict")
    plt.ylabel("Real")
    wandb.log({"matriz_confusao_yawndd": wandb.Image(plt)})
    plt.close()

    cm_cross = confusion_matrix(y_true_cross, preds_cross)
    plt.figure()
    sns.heatmap(cm_cross, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix NTHUDD")
    plt.xlabel("Predict")
    plt.ylabel("Real")
    wandb.log({"matriz_confusao_nthudd": wandb.Image(plt)})
    plt.close()

    # ========== GRAD-CAM ==========
    try:
        print("\nGerando Grad-CAM para YawDD dataset...")
        logar_gradcam_final(melhor_modelo, test_ds, CLASS_NAMES, num_imagens=4)
    except Exception as e:
        print(f"Erro ao gerar Grad-CAM para YawDD: {e}")

    try:
        print("\nGerando Grad-CAM para NTHUDD dataset...")
        logar_gradcam_final(melhor_modelo, cross_val_ds, CLASS_NAMES, num_imagens=4)
    except Exception as e:
        print(f"Erro ao gerar Grad-CAM para NTHUDD: {e}")

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
        "cross_auc": auc_cross,
        "False Positives": ultimo_valor(history.history, "fp"),
        "False Negatives": ultimo_valor(history.history, "fn"),
        "True Positives": ultimo_valor(history.history, "tp"),
        "True Negatives": ultimo_valor(history.history, "tn"),
        "precision_at_recall_80": ultimo_valor(history.history, "precision_at_recall_for_80"),
        "specificity_at_sensitivity_80": ultimo_valor(history.history, "specificity_at_sensitivity_for_80")
    })

    # ========== PROMOÇÃO DO MODELO (SE APROVADO) ==========
    if loss_val < 0.3 and overfit_loss < 0.1:
        melhor_modelo.save("melhor_modelo_resnet101.keras")
        art = wandb.Artifact(
            "melhor_modelo_resnet101",
            type="model",
            metadata={"val_loss": loss_val, "val_auc": auc_val, "overfit": overfit_loss}
        )
        art.add_file("melhor_modelo_resnet101.keras")
        wandb.log_artifact(art)
        print("Modelo promovido para o repositório de modelos")
    else:
        print("Modelo descartado")

    wandb.finish()

    # ========== RESULTADOS FINAIS ==========
    print("\n========================================================")
    print(" Avaliação na base YawDD:")
    print(f"Acurácia: {acuracia_val:.4f} | Precisão: {precisao:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
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