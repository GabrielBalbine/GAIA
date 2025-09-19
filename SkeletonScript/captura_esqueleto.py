# captura_esqueleto.py (Versão com Tracker Embutido do YOLO)

import cv2
import mediapipe as mp
import sys
import os
from ultralytics import YOLO

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print(">>> Carregando modelo de detecção e tracking (YOLOv8s)...")
model = YOLO('yolov8s.pt') 

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_finalissimo.mp4"
CONF_MINIMA_YOLO = 0.50

# ---------------- VERIFICAÇÃO E ABERTURA DO VÍDEO ----------------
if not os.path.exists(input_video_path): sys.exit(f"ERRO: Vídeo não encontrado: '{input_video_path}'")
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened(): sys.exit(f"ERRO: Não foi possível abrir o vídeo.")
frame_width, frame_height, fps = (int(cap.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
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
        
        # --- A MÁGICA ACONTECE AQUI ---
        # Usamos model.track() em vez de model(). classes=[0] para focar só em 'pessoa'.
        # 'persist=True' diz ao tracker para lembrar das pessoas entre os frames.
        results_yolo = model.track(image, persist=True, verbose=False, classes=[0])

        # Verifica se existem caixas de rastreamento no resultado
        if results_yolo[0].boxes.id is not None:
            # Pega as caixas e os IDs de rastreamento
            boxes = results_yolo[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results_yolo[0].boxes.id.cpu().numpy().astype(int)

            # Itera sobre cada pessoa rastreada
            for box, track_id in zip(boxes, track_ids):
                box_x1, box_y1, box_x2, box_y2 = box

                # Recorta a imagem da pessoa (ROI)
                roi = image[box_y1:box_y2, box_x1:box_x2]
                if roi.size == 0: continue

                # Roda o MediaPipe Pose SÓ no recorte
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(roi_rgb)
                
                # Desenha o esqueleto se encontrado
                if results_pose.pose_landmarks:
                    # Desenha o esqueleto direto no recorte para maior precisão
                    mp_drawing.draw_landmarks(roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # O desenho feito no 'roi' se reflete na 'image' original
                    cv2.putText(image, f"ID: {track_id}", (box_x1, box_y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        out.write(image)

# ------------------- FINALIZAÇÃO ---------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("-" * 30)
print(f">>> Processamento concluído!")
print(f">>> Vídeo salvo em: '{output_video_path}'")
print("-" * 30)