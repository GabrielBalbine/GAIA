# captura_esqueleto.py (Versão Final com Detector YOLO)

import cv2
import mediapipe as mp
import sys
import os
from ultralytics import YOLO

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Carrega o modelo de detecção de objetos YOLOv8
# Ele vai baixar os pesos automaticamente na primeira vez
print(">>> Carregando modelo de detecção de pessoas (YOLOv8)...")
model = YOLO('yolov8n.pt') # 'n' é o modelo "nano", o mais leve e rápido

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_yolo.mp4"
CONF_MINIMA = 0.50 # Confiança mínima para o YOLO

# ---------------- VERIFICAÇÃO E ABERTURA DO VÍDEO ----------------
if not os.path.exists(input_video_path):
    print(f"ERRO: Vídeo não encontrado: '{input_video_path}'")
    sys.exit(1)
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

with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(">>> Fim do vídeo.")
            break

        frame_count += 1
        
        # ETAPA 1: Detectar todas as PESSOAS no frame com o modelo YOLO
        results_yolo = model(image, verbose=False) # verbose=False para um log mais limpo

        # ETAPA 2: Para cada pessoa encontrada, rodar a estimação de pose
        # O resultado do YOLO já vem com os retângulos (boxes)
        for box in results_yolo[0].boxes:
            # A classe '0' no dataset COCO (usado pelo YOLO) é 'person'
            if box.cls == 0 and box.conf > CONF_MINIMA:
                # Pega as coordenadas do retângulo
                coords = box.xyxy[0].tolist()
                box_x1, box_y1, box_x2, box_y2 = map(int, coords)

                # Recorta a imagem da pessoa
                roi = image[box_y1:box_y2, box_x1:box_x2]
                if roi.size == 0: continue

                # Roda o MediaPipe Pose SÓ no recorte
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(roi_rgb)
                
                # Desenha o esqueleto se encontrado
                if results_pose.pose_landmarks:
                    # Converte os landmarks de volta para as coordenadas da imagem original
                    for landmark in results_pose.pose_landmarks.landmark:
                        landmark.x = (landmark.x * (box_x2 - box_x1) + box_x1) / frame_width
                        landmark.y = (landmark.y * (box_y2 - box_y1) + box_y1) / frame_height
                    
                    mp_drawing.draw_landmarks(
                        image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        out.write(image)

# ------------------- FINALIZAÇÃO ---------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f">>> Processamento concluído!")
print(f">>> Vídeo salvo em: '{output_video_path}'")
print("-" * 30)