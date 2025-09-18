# captura_esqueleto.py (Versão com Tracking para Estabilidade)

import cv2
import mediapipe as mp
import sys
import os
from ultralytics import YOLO

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print(">>> Carregando modelo de detecção de pessoas (YOLOv8)...")
model = YOLO('yolov8n.pt')

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_tracked.mp4"
CONF_MINIMA_YOLO = 0.50

# --- CONFIGURAÇÃO DO TRACKER ---
RE_DETECCAO_FRAME = 10 # A cada quantos frames vamos rodar o YOLO de novo
trackers = [] # Lista para armazenar nossos "mini-agentes" rastreadores

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
        
        # LÓGICA DE TRACKING E RE-DETECÇÃO
        # Se for o primeiro frame ou a cada X frames, rodamos o YOLO
        if frame_count % RE_DETECCAO_FRAME == 1:
            print(f"--- Rodando detecção completa no frame #{frame_count} ---")
            results_yolo = model(image, verbose=False)
            trackers.clear() # Limpa os trackers antigos
            
            for box in results_yolo[0].boxes:
                if box.cls == 0 and box.conf > CONF_MINIMA_YOLO:
                    coords = box.xyxy[0].tolist()
                    box_rect = (int(coords[0]), int(coords[1]), int(coords[2] - coords[0]), int(coords[3] - coords[1]))
                    
                    # Inicializa um novo tracker para cada pessoa encontrada
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(image, box_rect)
                    trackers.append(tracker)
        else:
            # Nos outros frames, apenas atualizamos a posição dos trackers
            new_trackers = []
            for tracker in trackers:
                success, box_rect = tracker.update(image)
                if success:
                    new_trackers.append(tracker) # Mantém só os que não se perderam
            trackers = new_trackers
            
        # Agora, para cada tracker ativo, fazemos a análise de pose
        for tracker in trackers:
            success, box_rect = tracker.update(image)
            if success:
                box_x1, box_y1, w, h = [int(v) for v in box_rect]
                box_x2, box_y2 = box_x1 + w, box_y1 + h
                
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
                        landmark.x = (landmark.x * w + box_x1) / frame_width
                        landmark.y = (landmark.y * h + box_y1) / frame_height
                    
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