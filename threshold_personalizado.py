import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import random
# ==========================================
# CONFIGURAÇÕES GLOBAIS
# ==========================================
CALIBRATION_SECONDS = 5          # tempo de calibração inicial
FPS = 30                         # assumindo 30 fps 
CALIBRATION_FRAMES = FPS * CALIBRATION_SECONDS

# Fatores para cálculo dos thresholds personalizados

EAR_THRESHOLD_FACTOR = 0.7       # EAR_thr = baseline_EAR * 0.7 (olho mais fechado que o normal)
MAR_THRESHOLD_FACTOR = 1.5       # MAR_thr = baseline_MAR * 1.5 (boca mais aberta que o normal)
# desvios padrão (ex: EAR_thr = baseline_EAR - 2*std_ear)

# Janela de suavização para decisão final
WINDOW_SIZE = 15
MIN_POSITIVE_RATIO = 0.6         # se mais de 60% dos frames na janela são positivos

# ==========================================
# MEDIAPIPE INIT
# ==========================================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,        # para landmarks dos olhos mais precisos
    min_detection_confidence=0.5
)

# Índices dos landmarks 
LEFT_EYE = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 386, 385, 263, 380, 374]


# ==========================================
# FUNÇÕES AUXILIARES

# ==========================================
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def EAR(lm, w, h, eye_indices):
    """Eye Aspect Ratio para um olho."""
    pts = [(lm[i].x * w, lm[i].y * h) for i in eye_indices]
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def MAR(lm, w, h):
    """Mouth Aspect Ratio."""
    top_lip = (lm[13].x * w, lm[13].y * h)      # lábio superior centro
    bottom_lip = (lm[14].x * w, lm[14].y * h)    # lábio inferior centro
    left_corner = (lm[61].x * w, lm[61].y * h)   # canto esquerdo
    right_corner = (lm[291].x * w, lm[291].y * h) # canto direito

    mouth_height = dist(top_lip, bottom_lip)
    mouth_width = dist(left_corner, right_corner)
    return mouth_height / (mouth_width + 1e-6)

def clahe(img):
    """Melhora o contraste da imagem."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# ==========================================
# CLASSE DE DETECÇÃO COM CALIBRAÇÃO PERSONALIZADA
# ==========================================

class DrowsinessDetector:
    def __init__(self, calib_frames=CALIBRATION_FRAMES):
        self.calib_frames = calib_frames
        self.calibrated = False
        self.ear_baseline = None
        self.mar_baseline = None
        self.ear_std = None
        self.mar_std = None
        self.ear_threshold = None
        self.mar_threshold = None

        # Buffers para calibração
        self.calib_ear = []
        self.calib_mar = []

        # Histórico para suavização
        self.ear_history = deque(maxlen=WINDOW_SIZE)
        self.mar_history = deque(maxlen=WINDOW_SIZE)
        self.pred_history = deque(maxlen=WINDOW_SIZE)

    def calibracao(self, ear, mar):
        """Coleta frames durante a calibração."""
        self.calib_ear.append(ear)
        self.calib_mar.append(mar)

        if len(self.calib_ear) >= self.calib_frames:
            # Calcula baseline (média) e desvio padrão
            self.ear_baseline = np.mean(self.calib_ear)
            self.mar_baseline = np.mean(self.calib_mar)
            self.ear_std = np.std(self.calib_ear)
            self.mar_std = np.std(self.calib_mar)

            #usar desvios padrão 
            self.ear_threshold = self.ear_baseline - 2 * self.ear_std
            self.mar_threshold = self.mar_baseline + 2 * self.mar_std

            # Garantir limites mínimos/máximos 
            self.ear_threshold = max(0.1, min(0.4, self.ear_threshold))
            self.mar_threshold = max(0.1, min(0.8, self.mar_threshold))

            self.calibrated = True
            print("="*50)
            print("CALIBRAÇÃO CONCLUÍDA!")
            print(f"EAR médio: {self.ear_baseline:.3f} (std: {self.ear_std:.3f})")
            print(f"MAR médio: {self.mar_baseline:.3f} (std: {self.mar_std:.3f})")
            print(f"Threshold EAR personalizado: {self.ear_threshold:.3f}")
            print(f"Threshold MAR personalizado: {self.mar_threshold:.3f}")
            print("="*50)
            return True
        return False

    def deteccao(self, ear, mar):
        """Classifica o frame atual usando thresholds personalizados e histórico."""
        if not self.calibrated:
            return None, 0.0   # ainda em calibração

        # Atualiza históricos suavizados
        self.ear_history.append(ear)
        self.mar_history.append(mar)

        ear_smooth = np.mean(self.ear_history)
        mar_smooth = np.mean(self.mar_history)

        # Aplica thresholds personalizados
        eyes_closed = ear_smooth < self.ear_threshold
        yawning = mar_smooth > self.mar_threshold

        # Decisão com níveis de confiança
        if eyes_closed and yawning:
            confidence = 0.9
            drowsy = True
        elif eyes_closed or yawning:
            confidence = 0.6
            drowsy = True
        else:
            confidence = 0.1
            drowsy = False

        # Adiciona ao histórico de predições
        self.pred_history.append(drowsy)

        # Filtro temporal: decisão baseada na maioria da janela
        if len(self.pred_history) >= WINDOW_SIZE:
            positive_ratio = sum(self.pred_history) / WINDOW_SIZE
            if positive_ratio >= MIN_POSITIVE_RATIO:
                final_drowsy = True
                final_conf = confidence  # pode ser a média das confianças, mas simplificamos
            elif positive_ratio <= (1 - MIN_POSITIVE_RATIO):
                final_drowsy = False
                final_conf = 1 - confidence
            else:
                # Mantém estado anterior (última predição)
                final_drowsy = drowsy
                final_conf = confidence
            return final_drowsy, final_conf
        else:
            return drowsy, confidence

# ==========================================
# FUNÇÃO PRINCIPAL (TEMPO REAL)
# ==========================================
def deteccao_em_tempo_real(source=0):
    """
    Executa detecção em tempo real.
    source: 0 para webcam, ou caminho de arquivo de vídeo.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erro ao abrir a fonte de vídeo.")
        return

    detector = DrowsinessDetector(calib_frames=CALIBRATION_FRAMES)
    calibrated = False
    frame_count = 0

    print("Iniciando captura. Mantenha-se em estado normal (olhos abertos, boca fechada) para calibração.")
    print(f"Calibração durará aproximadamente {CALIBRATION_SECONDS} segundos...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = clahe(frame)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # Se não detectar face, continua sem processar
            cv2.putText(frame, "Nenhum rosto detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        # Calcula EAR e MAR
        ear_left = EAR(landmarks, w, h, LEFT_EYE)
        ear_right = EAR(landmarks, w, h, RIGHT_EYE)
        ear = (ear_left + ear_right) / 2.0
        mar = MAR(landmarks, w, h)

        # Se ainda não calibrou
        if not detector.calibrated:
            calibrated = detector.calibrate(ear, mar)
            progress = min(100, int(len(detector.calib_ear) / CALIBRATION_FRAMES * 100))
            cv2.putText(frame, f"Calibrando... {progress}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Mantenha-se normal", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Detecção
            drowsy, conf = detector.detect(ear, mar)

            # Define cor e texto
            if drowsy:
                label = "SONOLENTO"
                color = (0, 0, 255)  # vermelho
            else:
                label = "ALERTA"
                color = (0, 255, 0)  # verde

            # Exibe informações no frame
            cv2.putText(frame, f"EAR: {ear:.3f} (thr={detector.ear_threshold:.3f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"MAR: {mar:.3f} (thr={detector.mar_threshold:.3f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Status: {label}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# GERAR DATASET DOS VÍDEO
# ==========================================
def pegar_id(video_path):
    nome = os.path.basename(video_path)
    return nome.split("-")[0]

def gerar_dataset(video_paths, output_dir="dataset_personalizado"):

    os.makedirs(output_dir, exist_ok=True)

    for subset in ["train","val","test"]:
        for label in ["sonolento","nao_sonolento"]:
            os.makedirs(os.path.join(output_dir, subset, label), exist_ok=True)

    # --------------------------------
    # AGRUPAR VIDEOS POR PESSOA
    # --------------------------------
    sujeitos = {}

    for video_path in video_paths:
        subject = pegar_id(video_path)
        sujeitos.setdefault(subject, []).append(video_path)

    ids_sujeitos = list(sujeitos.keys())
    random.shuffle(ids_sujeitos)

    n = len(ids_sujeitos)

    train_ids = ids_sujeitos[:int(0.7*n)]
    val_ids   = ids_sujeitos[int(0.7*n):int(0.85*n)]
    test_ids  = ids_sujeitos[int(0.85*n):]

    print("Train:", train_ids)
    print("Val:", val_ids)
    print("Test:", test_ids)

    # --------------------------------
    # PROCESSAR VIDEOS
    # --------------------------------
    for sujeito in sujeitos:

        if sujeito in train_ids:
            subset = "train"
        elif sujeito in val_ids:
            subset = "val"
        else:
            subset = "test"

        for video_path in sujeitos[sujeito]:

            video_name = os.path.splitext(os.path.basename(video_path))[0].lower()
            is_yawning_video = "yawning" in video_name
            folder_parts = os.path.normpath(video_path).lower().split(os.sep)
            is_dash = "dash" in folder_parts

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            # --------------------------------
            # CALIBRAÇÃO
            # --------------------------------
            calib_ear = []
            calib_mar = []

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30

            calib_frames = int(fps * CALIBRATION_SECONDS)

            while len(calib_ear) < calib_frames:

                ret, frame = cap.read()
                if not ret:
                    break

                frame = clahe(frame)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    continue

                landmarks = results.multi_face_landmarks[0].landmark

                ear = (EAR(landmarks, w, h, LEFT_EYE) + EAR(landmarks, w, h, RIGHT_EYE)) / 2
                mar = MAR(landmarks, w, h)

                calib_ear.append(ear)
                calib_mar.append(mar)

            if len(calib_ear) == 0:
                print(f"Sem dados de calibração para {video_path}")
                cap.release()
                continue

            # --------------------------------
            # THRESHOLDS PERSONALIZADOS
            # --------------------------------
            
            k = 2 # nível de sensibilidade (quantos desvios padrão para definir o threshold)
            
            ear_baseline = np.mean(calib_ear)
            mar_baseline = np.mean(calib_mar)

            ear_std = np.std(calib_ear)
            mar_std = np.std(calib_mar)

            ear_thr = ear_baseline - k * ear_std
            mar_thr = mar_baseline + k * mar_std
            
            # clamp para evitar valores absurdos
            ear_thr = max(0.1, min(0.4, ear_thr))
            mar_thr = max(0.1, min(0.8, mar_thr))

            print(f"{video_name} -> EAR_thr={ear_thr:.3f} (std={ear_std:.3f}) | MAR_thr={mar_thr:.3f} (std={mar_std:.3f})")

           
            # --------------------------------
            # PROCESSAR FRAMES
            # --------------------------------
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            frame_id = 0

            while True:

                ret, frame = cap.read()
                if not ret:
                    break

                frame = clahe(frame)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    continue

                landmarks = results.multi_face_landmarks[0].landmark

                ear = (EAR(landmarks, w, h, LEFT_EYE) + EAR(landmarks, w, h, RIGHT_EYE)) / 2
                mar = MAR(landmarks, w, h)

                # --------------------------------
                # CLASSIFICAÇÃO
                # --------------------------------
                if is_yawning_video:
                    if mar > mar_thr or ear < ear_thr:
                        label = "sonolento"
                    else:
                        label = "nao_sonolento"

                elif is_dash:
                    if mar > mar_thr:
                        label = "sonolento"
                    else:
                        label = "nao_sonolento"

                else:
                    label = "nao_sonolento"

                out_dir = os.path.join(output_dir, subset, label)

                filename = f"{video_name}_frame_{frame_id:05d}_EAR_{ear:.3f}_MAR_{mar:.3f}.jpg"

                cv2.imwrite(os.path.join(out_dir, filename), frame)

                frame_id += 1

            cap.release()

            print(f"{video_name}: EAR {ear:.3f} MAR {mar:.3f} -> {frame_id} frames processados")

def balancear_dataset(dataset_path):

    subsets = ["train", "val", "test"]

    for subset in subsets:

        sonolento_path = os.path.join(dataset_path, subset, "sonolento")
        nao_path = os.path.join(dataset_path, subset, "nao_sonolento")

        sonolento_files = os.listdir(sonolento_path)
        nao_files = os.listdir(nao_path)

        n_son = len(sonolento_files)
        n_nao = len(nao_files)

        print(f"\n{subset}")
        print(f"antes -> sonolento: {n_son} | nao_sonolento: {n_nao}")

        # queremos que nao_sonolento tenha o mesmo número que sonolento
        if n_nao > n_son:

            remover = random.sample(nao_files, n_nao - n_son)

            for file in remover:
                os.remove(os.path.join(nao_path, file))

        elif n_son > n_nao:

            remover = random.sample(sonolento_files, n_son - n_nao)

            for file in remover:
                os.remove(os.path.join(sonolento_path, file))

        print(f"depois -> {min(n_son,n_nao)} cada classe")
# ==========================================
# EXEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    # Escolha o modo:
    # 1 - Tempo real com webcam (thresholds personalizados por calibração)
    # 2 - Gerar dataset com thresholds personalizados por vídeo

    MODO = 2  # Altere para 2 se quiser gerar dataset

    if MODO == 1:
        # Webcam (0) ou arquivo de vídeo (ex: "video.avi")
        deteccao_em_tempo_real(source=0)

    elif MODO == 2:
        # Lista de vídeos para gerar dataset (cada vídeo terá seus thresholds)
        video_list = [
        r"C:\Users\Dell\Downloads\data\YawDD dataset\Mirror\Male_mirror",
        r"C:\Users\Dell\Downloads\data\YawDD dataset\Mirror\Female_mirror",
        r"C:\Users\Dell\Downloads\data\YawDD dataset\Dash\Male",
        r"C:\Users\Dell\Downloads\data\YawDD dataset\Dash\Female"
            ]
        # Coletar todos os arquivos .avi dentro das pastas
        video_files = []
        for folder in video_list:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith(".avi"):
                        video_files.append(os.path.join(folder, file))
            else:
                print(f"Pasta não encontrada: {folder}")

        print(f"Encontrados {len(video_files)} vídeos.")
        gerar_dataset(video_files, output_dir="dataset_personalizado")
        balancear_dataset("dataset_personalizado")