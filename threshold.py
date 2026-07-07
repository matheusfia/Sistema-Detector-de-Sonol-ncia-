import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
from scipy.stats import norm
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
import json

# ==========================================
# CONFIGURAÇÕES
# ==========================================

DATASET_PATHS = [
    r"C:\Users\Dell\Downloads\data\YawDD dataset\Mirror\Male_mirror",
    r"C:\Users\Dell\Downloads\data\YawDD dataset\Mirror\Female_mirror",
    r"C:\Users\Dell\Downloads\data\YawDD dataset\Dash\Male",
    r"C:\Users\Dell\Downloads\data\YawDD dataset\Dash\Female"
]

OUTPUT_DIR = "dataset_final"
FPS = 30
BASELINE_SECONDS = 5
BASELINE_FRAMES = FPS * BASELINE_SECONDS

# Parâmetros ajustados
WINDOW = 15
MIN_EYE_CLOSE_FRAMES = 8
EMA_ALPHA = 0.2

# Thresholds manuais baseados na literatura (mais confiáveis)
MANUAL_EAR_THR = 0.15  # Valor típico para olhos fechados
MANUAL_MAR_THR = 0.5   # Valor típico para bocejo

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MEDIAPIPE
# ==========================================

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Índices dos landmarks (CORRETOS para o MediaPipe)
LEFT_EYE = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 386, 385, 263, 380, 374]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# ==========================================
# FUNÇÕES CORRIGIDAS
# ==========================================

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def EAR(lm, w, h, eye):
    """Eye Aspect Ratio - CORRETO"""
    pts = [(lm[i].x * w, lm[i].y * h) for i in eye]
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def MAR_CORRETO(lm, w, h):
    """Mouth Aspect Ratio - VERSÃO FINAL CORRIGIDA"""
    
    # Pontos principais para boca (MediaPipe)
    # Superior e inferior da boca (centro)
    top_lip = (lm[13].x * w, lm[13].y * h)      # Lábio superior (centro)
    bottom_lip = (lm[14].x * w, lm[14].y * h)    # Lábio inferior (centro)
    
    # Cantos da boca
    left_corner = (lm[61].x * w, lm[61].y * h)   # Canto esquerdo
    right_corner = (lm[291].x * w, lm[291].y * h) # Canto direito
    
    # Calcular altura e largura
    mouth_height = dist(top_lip, bottom_lip)
    mouth_width = dist(left_corner, right_corner)
    
    # MAR = altura / largura
    mar = mouth_height / (mouth_width + 1e-6)
    
    return mar

def MAR_SEM_BOCEJO(lm, w, h):
    """MAR apenas para boca fechada (filtro)"""
    mar = MAR_CORRETO(lm, w, h)
    
    # Se MAR muito baixo, provavelmente boca fechada
    if mar < 0.15:
        return 0.0  # Indicador de boca fechada
    
    return mar

def clahe(img):
    """Melhorar contraste"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def detect_yawn(mar_value, threshold=0.25, min_duration=5):
    """
    Detecção de bocejo baseada em MAR com persistência temporal
    """
    # Usar threshold mais baixo para capturar aberturas parciais
    return mar_value > threshold

# ==========================================
# FUNÇÃO PARA ANÁLISE DETALHADA DOS DADOS
# ==========================================

def analyze_distributions(ear_normal, ear_drowsy, mar_normal, mar_drowsy):
    """Análise completa das distribuições"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Histogramas EAR
    axes[0, 0].hist(ear_normal, bins=50, alpha=0.5, label='Normal', density=True)
    axes[0, 0].hist(ear_drowsy, bins=50, alpha=0.5, label='Drowsy', density=True)
    axes[0, 0].axvline(x=0.25, color='r', linestyle='--', label='Threshold (0.25)')
    axes[0, 0].set_xlabel('EAR')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution EAR')
    axes[0, 0].legend()
    
    # 2. Histogramas MAR
    axes[0, 1].hist(mar_normal, bins=50, alpha=0.5, label='Normal', density=True)
    axes[0, 1].hist(mar_drowsy, bins=50, alpha=0.5, label='Drowsy', density=True)
    axes[0, 1].axvline(x=0.25, color='r', linestyle='--', label='Threshold (0.25)')
    axes[0, 1].set_xlabel('MAR')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution MAR')
    axes[0, 1].legend()
    
    # 3. Boxplots
    axes[0, 2].boxplot([ear_normal, ear_drowsy], labels=['Normal', 'Drowsy'])
    axes[0, 2].set_ylabel('EAR')
    axes[0, 2].set_title('Boxplot EAR')
    axes[0, 2].axhline(y=0.25, color='r', linestyle='--', label='Threshold')
    axes[0, 2].legend()
    
    # 4. Curvas Precision-Recall EAR
    y_true_ear = np.concatenate([np.zeros(len(ear_normal)), np.ones(len(ear_drowsy))])
    ear_all = np.concatenate([ear_normal, ear_drowsy])
    
    thresholds_ear = np.linspace(0.1, 0.4, 50)
    f1_scores_ear = []
    
    for thr in thresholds_ear:
        pred = (ear_all < thr).astype(int)
        f1 = f1_score(y_true_ear, pred, zero_division=0)
        f1_scores_ear.append(f1)
    
    axes[1, 0].plot(thresholds_ear, f1_scores_ear, 'b-', linewidth=2)
    axes[1, 0].axvline(x=0.25, color='r', linestyle='--', label='Manual Threshold')
    best_ear_idx = np.argmax(f1_scores_ear)
    axes[1, 0].axvline(x=thresholds_ear[best_ear_idx], color='g', linestyle='--', 
                       label=f'OptimalThreshold: {thresholds_ear[best_ear_idx]:.3f}')
    axes[1, 0].set_xlabel('Threshold EAR')
    axes[1, 0].set_ylabel('F1-score')
    axes[1, 0].set_title('F1-score vs Threshold EAR')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Curvas Precision-Recall MAR
    y_true_mar = np.concatenate([np.zeros(len(mar_normal)), np.ones(len(mar_drowsy))])
    mar_all = np.concatenate([mar_normal, mar_drowsy])
    
    thresholds_mar = np.linspace(0.05, 0.4, 50)
    f1_scores_mar = []
    
    for thr in thresholds_mar:
        pred = (mar_all > thr).astype(int)
        f1 = f1_score(y_true_mar, pred, zero_division=0)
        f1_scores_mar.append(f1)
    
    axes[1, 1].plot(thresholds_mar, f1_scores_mar, 'g-', linewidth=2)
    axes[1, 1].axvline(x=0.25, color='r', linestyle='--', label='Manual Threshold')
    best_mar_idx = np.argmax(f1_scores_mar)
    axes[1, 1].axvline(x=thresholds_mar[best_mar_idx], color='g', linestyle='--',
                       label=f'Optimal Threshold: {thresholds_mar[best_mar_idx]:.3f}')
    axes[1, 1].set_xlabel('Threshold MAR')
    axes[1, 1].set_ylabel('F1-score')
    axes[1, 1].set_title('F1-score vs Threshold MAR')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Scatter plot EAR vs MAR
    axes[1, 2].scatter(ear_normal, mar_normal, alpha=0.3, label='Normal', s=1)
    axes[1, 2].scatter(ear_drowsy, mar_drowsy, alpha=0.3, label='Drowsy', s=1)
    axes[1, 2].axvline(x=0.25, color='r', linestyle='--', label='EAR Thr')
    axes[1, 2].axhline(y=0.25, color='r', linestyle='--', label='MAR Thr')
    axes[1, 2].set_xlabel('EAR')
    axes[1, 2].set_ylabel('MAR')
    axes[1, 2].set_title('EAR vs MAR')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('analise_completa_distribuicoes.png', dpi=150)
    plt.show()
    
    return thresholds_ear[best_ear_idx], thresholds_mar[best_mar_idx]

# ==========================================
# COLETAR DADOS
# ==========================================

print("📊 Coletando dados para análise...")

ear_normal = []
ear_drowsy = []
mar_normal = []
mar_drowsy = []

# Coletar apenas de alguns vídeos para análise
videos_amostra = []
for root in DATASET_PATHS:
    for f in os.listdir(root)[:10]:  # 10 vídeos de cada pasta
        if f.endswith(".avi"):
            videos_amostra.append(os.path.join(root, f))

for video in videos_amostra:
    name = os.path.basename(video).replace(".avi", "")
    is_drowsy = 1 if "yawning" in name.lower() else 0
    
    cap = cv2.VideoCapture(video)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = clahe(frame)
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        
        if not res.multi_face_landmarks:
            continue
        
        lm = res.multi_face_landmarks[0].landmark
        
        ear = (EAR(lm, w, h, LEFT_EYE) + EAR(lm, w, h, RIGHT_EYE)) / 2
        mar = MAR_CORRETO(lm, w, h)
        
        if is_drowsy:
            ear_drowsy.append(ear)
            mar_drowsy.append(mar)
        else:
            ear_normal.append(ear)
            mar_normal.append(mar)
    
    cap.release()

# Converter para numpy
ear_normal = np.array(ear_normal)
ear_drowsy = np.array(ear_drowsy)
mar_normal = np.array(mar_normal)
mar_drowsy = np.array(mar_drowsy)

# ==========================================
# ANÁLISE E THRESHOLDS OTIMIZADOS
# ==========================================

print("\n📊 ESTATÍSTICAS:")
print(f"EAR - Normal: μ={np.mean(ear_normal):.4f}, σ={np.std(ear_normal):.4f}")
print(f"EAR - Drowsy: μ={np.mean(ear_drowsy):.4f}, σ={np.std(ear_drowsy):.4f}")
print(f"MAR - Normal: μ={np.mean(mar_normal):.4f}, σ={np.std(mar_normal):.4f}")
print(f"MAR - Drowsy: μ={np.mean(mar_drowsy):.4f}, σ={np.std(mar_drowsy):.4f}")

# Análise completa
best_ear, best_mar = analyze_distributions(ear_normal, ear_drowsy, mar_normal, mar_drowsy)

print(f"\n🎯 Thresholds Otimizados por F1-score:")
print(f"EAR Threshold: {best_ear:.4f}")
print(f"MAR Threshold: {best_mar:.4f}")

# ==========================================
# CLASSIFICAÇÃO MELHORADA
# ==========================================

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, mar_threshold=0.25):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)
        self.pred_history = deque(maxlen=15)
        
    def detect(self, ear, mar):
        """Detecção com contexto temporal"""
        
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        
        # Suavização
        ear_smooth = np.mean(self.ear_history)
        mar_smooth = np.mean(self.mar_history)
        
        # Condições
        eyes_closed = ear_smooth < self.ear_threshold
        yawning = mar_smooth > self.mar_threshold
        
        # Regras de decisão
        if eyes_closed and yawning:
            confidence = 0.9
            drowsy = True
        elif eyes_closed or yawning:
            confidence = 0.6
            drowsy = True
        else:
            confidence = 0.1
            drowsy = False
        
        # Persistência temporal (evita flickering)
        self.pred_history.append(drowsy)
        
        if len(self.pred_history) > 5:
            # Se maioria nos últimos frames
            if sum(self.pred_history) > len(self.pred_history) * 0.6:
                return True, confidence
            elif sum(self.pred_history) < len(self.pred_history) * 0.4:
                return False, confidence
        
        return drowsy, confidence

# ==========================================
# GERAR DATASET COM THRESHOLDS OTIMIZADOS
# ==========================================

print("\n🚀 Gerando dataset com thresholds otimizados...")

detector = DrowsinessDetector(ear_threshold=best_ear, mar_threshold=best_mar)

for video in videos_amostra:
    name = os.path.basename(video).replace(".avi", "")
    pid, atype = name.split("-")
    
    cap = cv2.VideoCapture(video)
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = clahe(frame)
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        
        if not res.multi_face_landmarks:
            continue
        
        lm = res.multi_face_landmarks[0].landmark
        
        ear = (EAR(lm, w, h, LEFT_EYE) + EAR(lm, w, h, RIGHT_EYE)) / 2
        mar = MAR_CORRETO(lm, w, h)
        
        # Detectar sonolência
        drowsy, confidence = detector.detect(ear, mar)
        
        # Salvar frame com anotações
        label = "sonolento" if drowsy else "nao_sonolento"
        
        # Adicionar informações no frame
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {label}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if drowsy else (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Salvar
        save_dir = os.path.join(OUTPUT_DIR, "train" if frame_id < 100 else "val", label)
        os.makedirs(save_dir, exist_ok=True)
        
        fname = f"{pid}_{atype}_EAR_{ear:.3f}_MAR_{mar:.3f}_frame_{frame_id}.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), frame)
        
        frame_id += 1
    
    cap.release()

print(f"\n✅ Dataset gerado em: {OUTPUT_DIR}")
print(f"📊 Thresholds utilizados: EAR={best_ear:.3f}, MAR={best_mar:.3f}")