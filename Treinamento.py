

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import wandb
from wandb.integration.keras import WandbCallback
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# ==============================================================  
# 1️⃣ PARÂMETROS
# ==============================================================  
base_dir = "../drowsiness_system/dataset"
IMG_SIZE = (128,128)
BATCH_SIZE = 32
checkpoint_path = "melhor_modelo.keras"

# ==============================================================  
# 2️⃣ INICIALIZAÇÃO W&B
# ==============================================================  
wandb.init(
    project="drowsiness_detection_cnn_advanced",
    config={
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", "precision", "recall"],
        "epochs": 30
    }
)
config = wandb.config

# ==============================================================  
# 3️⃣ DATASET
# ==============================================================  
train_ds = image_dataset_from_directory(
    f"{base_dir}/train", 
    labels="inferred", 
    label_mode="int", 
    batch_size=BATCH_SIZE, 
    image_size=IMG_SIZE,
    shuffle=True
    )
val_ds = image_dataset_from_directory(
    f"{base_dir}/val", 
    labels="inferred", 
    label_mode="int", 
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE, 
    shuffle=True
    )
test_ds = image_dataset_from_directory(
    f"{base_dir}/test",
    labels="inferred", 
    label_mode="int", 
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE, 
    shuffle=False
    )

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================================================  
# 4️⃣ NORMALIZAÇÃO E AUMENTO DE DADOS
# ==============================================================  

normalization_layer = keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))



data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1)
])

# ==============================================================  
# 5️⃣ CNN
# ==============================================================  
model = Sequential([
    data_augmentation,
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)

# ==============================================================  
# 6️⃣ CALLBACKS
# ==============================================================  
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(
    monitor='val_loss',   # métrica monitorada
    factor=0.5,           # multiplica LR por 0.5 quando não melhora
    patience=3,           # espera 3 epochs antes de reduzir
    min_lr=1e-6,          # limite mínimo
    verbose=1
)
]

# ==============================================================  
# 7️⃣ TREINAMENTO
# ==============================================================  
history = model.fit(train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=callbacks )

# Tracking manual no W&B (opcional)
# for epoch in range(len(history.history['loss'])):
#     wandb.log({
#         "epoch": epoch+1,
#         "train_loss": history.history['loss'][epoch],
#         "val_loss": history.history['val_loss'][epoch],
#         "train_accuracy": history.history['accuracy'][epoch],
#         "val_accuracy": history.history['val_accuracy'][epoch]
#     })

# ==============================================================  
# 8️⃣ GRÁFICOS DE TREINAMENTO
# ==============================================================  
plt.figure(figsize=(12,5))

# Perda
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Curva de Perda')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Acurácia
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Curva de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()

# ==============================================================  
# 9️⃣ AVALIAÇÃO FINAL
# ==============================================================  
print("\n Avaliando o melhor modelo no conjunto de teste...")
melhor_modelo = load_model(checkpoint_path)
results = melhor_modelo.evaluate(test_ds)
print("Avaliação no conjunto de teste:")
print("Loss:", results[0])
print("Acurácia:", results[1])
print("Precisão:", results[2])
print("Recall:", results[3])



# Log final no W&B
wandb.log({"test_loss": results[0], "test_accuracy": results[1], "test_precision": results[2], "test_recall": results[3]})

# ==============================================================  
# 🔟 MATRIZ DE CONFUSÃO E ROC
# ==============================================================  
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = melhor_modelo.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["nao_sonolento", "sonolento"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()

# ROC e AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("Falsos Positivos (1 - Especificidade)")
plt.ylabel("Verdadeiros Positivos (Sensibilidade)")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()

# Converter de [p] → [[1-p, p]] para cada amostra
y_pred_probs_bin = np.concatenate([1 - y_pred_probs, y_pred_probs], axis=1)

# Cálculo das métricas
precisao = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Log manual para o W&B
wandb.log({
    "precision": precisao,
    "recall": recall,
    "f1_score": f1
})

wandb.log({
    "roc_auc": roc_auc,
    "confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                    y_true=y_true,
                                                    preds=y_pred,
                                                    class_names=["nao_sonolento", "sonolento"]),
    "pr_curve": wandb.plot.pr_curve(y_true, y_pred_probs_bin, labels=["nao_sonolento", "sonolento"])
})

print("\n📋 Relatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=["nao_sonolento", "sonolento"]))

# ==============================================================  
# 🔟 FUNÇÃO PARA TESTAR IMAGEM NOVA
# ==============================================================  
def prever_imagem(caminho_img):
    img = load_img(caminho_img, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob = melhor_modelo.predict(img_array)[0][0]
    classe = "sonolento 😴" if prob > 0.5 else "nao_sonolento 🙂"
    print(f"🔹 Previsão: {classe} (confiança: {prob:.2f})")
    return classe, prob