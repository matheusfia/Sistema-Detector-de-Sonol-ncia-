import cv2
import numpy as np
import mediapipe as mp
import math
import time
import winsound  # Para beep no Windows; em Linux use 'os.system("beep -f 1000")' ou similar

# -------------------------------
# Constantes e limiares
# -------------------------------
# Limiares para pose da cabeça (em graus)
FAIXA_RETA = 10          # ±10° para yaw (esquerda/direita) e pitch (cima/baixo)
MAX_CONTADOR_POSE = 30   # número de frames antes de disparar alarme

# Limiares para Razão de Aspecto do Olho (EAR)
EAR_LIMIAR = 0.2       # abaixo disso -> olhos fechados
FRAMES_OLHO_FECHADO = 30  # número de frames consecutivos com EAR < limiar para disparar

# Limiares para Razão de Aspecto da Boca (MAR)
MAR_LIMIAR = 0.6         # acima disso -> boca aberta (bocejo)
FRAMES_BOCEJO = 10        # número de frames consecutivos com MAR > limiar para disparar

# Calibração da câmera para solvePnP
# Distância focal aproximada para webcam comum; pode ser ajustada
def obter_matriz_camera(largura, altura):
    distancia_focal = largura  # aproximação
    centro = (largura / 2, altura / 2)
    matriz_camera = np.array([[distancia_focal, 0, centro[0]],
                              [0, distancia_focal, centro[1]],
                              [0, 0, 1]], dtype=np.float32)
    return matriz_camera

# Pontos 3D do modelo facial (em mm) para os 6 landmarks escolhidos
# Índices correspondentes aos landmarks do MediaPipe FaceMesh:
#   1  : ponta do nariz
#   33 : canto interno do olho esquerdo
#   263: canto interno do olho direito
#   61 : canto esquerdo da boca
#   291: canto direito da boca
#   199: queixo
# Valores baseados em um modelo médio de rosto
pontos_3d_modelo = np.array([
    (0.0, 0.0, 0.0),          # ponta do nariz
    (-30.0, -30.0, -50.0),    # olho esquerdo (canto interno)
    (30.0, -30.0, -50.0),     # olho direito (canto interno)
    (-30.0, -50.0, -80.0),    # canto esquerdo da boca
    (30.0, -50.0, -80.0),     # canto direito da boca
    (0.0, -80.0, -130.0)      # queixo
], dtype=np.float32)

# Índices no MediaPipe FaceMesh para os pontos acima
INDICES_LANDMARKS = [1, 33, 263, 61, 291, 199]

# -------------------------------
# Funções auxiliares
# -------------------------------
def obter_pontos_2d(landmarks, img_larg, img_alt):
    """Extrai as coordenadas 2D dos landmarks selecionados."""
    pontos_2d = []
    for idx in INDICES_LANDMARKS:
        lm = landmarks[idx]
        x, y = int(lm.x * img_larg), int(lm.y * img_alt)
        pontos_2d.append((x, y))
    return np.array(pontos_2d, dtype=np.float32)

def obter_angulos_euler(vetor_rotacao):
    """Converte vetor de rotação em ângulos de Euler (roll, pitch, yaw) em graus."""
    R, _ = cv2.Rodrigues(vetor_rotacao)
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

def razao_aspecto_olho(landmarks, img_larg, img_alt, indices):
    """Calcula EAR para um olho dado os 6 índices dos landmarks."""
    pontos = []
    for i in indices:
        lm = landmarks[i]
        pontos.append((lm.x * img_larg, lm.y * img_alt))
    pontos = np.array(pontos, dtype=np.float32)
    # Distâncias verticais
    v1 = np.linalg.norm(pontos[1] - pontos[5])
    v2 = np.linalg.norm(pontos[2] - pontos[4])
    # Distância horizontal
    h = np.linalg.norm(pontos[0] - pontos[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

def razao_aspecto_boca(landmarks, img_larg, img_alt):
    """Calcula MAR usando os landmarks externos da boca."""
    # Índices da boca no MediaPipe FaceMesh: 61 (esquerda), 291 (direita), 13 (superior), 14 (inferior)
    esquerda = (landmarks[61].x * img_larg, landmarks[61].y * img_alt)
    direita = (landmarks[291].x * img_larg, landmarks[291].y * img_alt)
    topo = (landmarks[13].x * img_larg, landmarks[13].y * img_alt)
    base = (landmarks[14].x * img_larg, landmarks[14].y * img_alt)
    dist_vert = np.linalg.norm(np.array(topo) - np.array(base))
    dist_horiz = np.linalg.norm(np.array(esquerda) - np.array(direita))
    mar = dist_vert / dist_horiz
    return mar

# -------------------------------
# Loop principal de detecção
# -------------------------------
def main():
    # Inicializa MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Captura de vídeo (0 = webcam padrão)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    # Obtém dimensões do frame para calibração da câmera
    ret, frame = cap.read()
    if not ret:
        print("Erro: não foi possível ler o frame.")
        return
    h, w = frame.shape[:2]
    matriz_camera = obter_matriz_camera(w, h)
    coeffs_dist = np.zeros((4,1))  # assume sem distorção

    # Variáveis de estado
    contador_pose = 0
    contador_olho_fechado = 0
    contador_bocejo = 0

    # Olho esquerdo: [canto_ext, sup_ext, sup_int, canto_int, inf_int, inf_ext]
    OLHO_ESQUERDO_INDICES = [33, 159, 158, 133, 153, 145]

    # Olho direito: [canto_ext, sup_ext, sup_int, canto_int, inf_int, inf_ext]
    OLHO_DIREITO_INDICES = [362, 386, 385, 263, 380, 374]

    alarme_ativado = False
    ultimo_alarme = 0

    print("Sistema de detecção de sonolência iniciado. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelha horizontalmente para visão de espelho (opcional)
        frame = cv2.flip(frame, 1)
  

        # APLICAÇÃO DO CLAHE PARA MELHORAR DETECÇÃO EM SOMBRAS
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Conversão para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Textos padrão para exibição
        status_cabeca = "Cabeca: Reta"
        status_olho = "Olhos: Abertos"
        status_boca = "Boca: Fechada"
        msg_alerta = ""

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # ----- Estimativa da pose da cabeça -----
            pontos_2d = obter_pontos_2d(landmarks, w, h)
            sucesso, vetor_rot, _ = cv2.solvePnP(
                pontos_3d_modelo, pontos_2d, matriz_camera, coeffs_dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if sucesso:
                roll, pitch, yaw = obter_angulos_euler(vetor_rot)
                # Classifica a pose conforme os limiares: pitch (cima/baixo) e yaw (esquerda/direita)
                if pitch > FAIXA_RETA:
                    status_cabeca = "Cabeca: Olhando para cima"
                elif pitch < -FAIXA_RETA:
                    status_cabeca = "Cabeca: Olhando para baixo"
                elif yaw > FAIXA_RETA:
                    status_cabeca = "Cabeca: Olhando para direita"
                elif yaw < -FAIXA_RETA:
                    status_cabeca = "Cabeca: Olhando para esquerda"
                else:
                    status_cabeca = "Cabeca: Reta"

                # Contador de distração: se pitch ou yaw saírem da faixa reta
                if abs(pitch) > FAIXA_RETA or abs(yaw) > FAIXA_RETA:
                    contador_pose += 1
                else:
                    contador_pose = max(0, contador_pose - 1)  # decaimento

                # Exibe ângulos na tela
                cv2.putText(frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ----- Razão de Aspecto do Olho (EAR) -----
            ear_esq = razao_aspecto_olho(landmarks, w, h, OLHO_ESQUERDO_INDICES)
            ear_dir = razao_aspecto_olho(landmarks, w, h, OLHO_DIREITO_INDICES)
            ear = (ear_esq + ear_dir) / 2.0
            if ear < EAR_LIMIAR:
                contador_olho_fechado += 1
                status_olho = "Olhos: Fechados"
            else:
                contador_olho_fechado = 0
                status_olho = "Olhos: Abertos"

            # ----- Razão de Aspecto da Boca (MAR) -----
            mar = razao_aspecto_boca(landmarks, w, h)
            if mar > MAR_LIMIAR:
                contador_bocejo += 1
                status_boca = "Boca: Bocejando"
            else:
                contador_bocejo = 0
                status_boca = "Boca: Fechada"

            # ----- Verifica condições de sonolência/distração -----
            if contador_pose >= MAX_CONTADOR_POSE:
                msg_alerta = "ALERTA: Distracao pela cabeca!"
            if contador_olho_fechado >= FRAMES_OLHO_FECHADO:
                msg_alerta = "ALERTA: Olhos fechados!"
            if contador_bocejo >= FRAMES_BOCEJO:
                msg_alerta = "ALERTA: Bocejo detectado!"

            # Dispara alarme sonoro se houver alerta
            if msg_alerta:
                agora = time.time()
                if agora - ultimo_alarme > 3:  # evita alarmes repetitivos
                    winsound.Beep(1000, 500)   # beep de 1000Hz por 500ms (Windows)
                    ultimo_alarme = agora
                # Reseta contadores para não alarmar continuamente no mesmo evento
                contador_pose = 0
                contador_olho_fechado = 0
                contador_bocejo = 0

        # Exibe informações no frame
        cv2.putText(frame, status_cabeca, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_olho, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_boca, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if msg_alerta:
            cv2.putText(frame, msg_alerta, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # Mostra o vídeo
        cv2.imshow("Detector de Sonolencia", frame)

        # Sai ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()