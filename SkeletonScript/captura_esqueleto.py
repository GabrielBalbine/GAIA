# captura_esqueleto.py (Versão Profissional: Detecta Faces e depois Estima a Pose)

import cv2
import mediapipe as mp
import sys
import os

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Carrega o detector de rostos pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_final.mp4"

# ---------------- VERIFICAÇÃO INICIAL ----------------
if not os.path.exists(input_video_path):
    print(f"ERRO: Vídeo não encontrado: '{input_video_path}'")
    sys.exit(1)

# ----------------- ABERTURA DO VÍDEO -----------------
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"ERRO: Não foi possível abrir o vídeo.")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f">>> Vídeo aberto! Resolução: {frame_width}x{frame_height}, FPS: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ---------------- LOOP DE PROCESSAMENTO ----------------
print(f">>> Processando vídeo '{input_video_path}'...")
frame_count = 0

with mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(">>> Fim do vídeo.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processando frame #{frame_count}...")
            
        # ETAPA 1: Detectar todos os rostos no frame
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

        lista_de_esqueletos = []
        # ETAPA 2: Para cada rosto encontrado, rodar a estimação de pose
        for (x, y, w, h) in faces:
            # Cria uma "Região de Interesse" (ROI) um pouco maior que o rosto
            # para dar contexto ao detector de pose.
            roi_x = max(0, x - w)
            roi_y = max(0, y - h)
            roi_w = min(frame_width - roi_x, w * 3)
            roi_h = min(frame_height - roi_y, h * 4)

            # Recorta a imagem
            roi = image[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            
            # Converte o recorte para RGB e processa com o MediaPipe
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = pose.process(roi_rgb)
            
            # Se um esqueleto foi encontrado no recorte, armazena-o
            if results.pose_landmarks:
                # IMPORTANTE: As coordenadas dos landmarks estão relativas ao recorte (ROI).
                # Precisamos convertê-las de volta para as coordenadas da imagem completa.
                landmarks_absolutos = results.pose_landmarks
                for landmark in landmarks_absolutos.landmark:
                    landmark.x = (landmark.x * roi_w + roi_x) / frame_width
                    landmark.y = (landmark.y * roi_h + roi_y) / frame_height
                lista_de_esqueletos.append(landmarks_absolutos)

        # Agora temos uma lista de esqueletos! Vamos desenhá-los.
        # (Aqui você pode reinserir a lógica de identificar_pessoas se quiser)
        for esqueleto in lista_de_esqueletos:
            mp_drawing.draw_landmarks(
                image, esqueleto, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        out.write(image)

# ------------------- FINALIZAÇÃO ---------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f">>> Processamento concluído!")
print(f">>> Total de {frame_count} frames processados.")
print(f">>> Vídeo salvo em: '{output_video_path}'")
print("-" * 30)