# captura_esqueleto.py (Versão com chamada do tracker.update() corrigida)

import cv2
import mediapipe as mp
import sys
import os
from ultralytics import YOLO
from sort import SortTracker
import numpy as np

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print(">>> Carregando modelo de detecção de pessoas (YOLOv8s)...")
model = YOLO('yolov8s.pt') 

tracker = SortTracker(max_age=20, min_hits=3, iou_threshold=0.3)

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_pro.mp4"
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

        results_yolo = model(image, verbose=False)

        detections_for_sort = []
        for box in results_yolo[0].boxes:
            if box.cls == 0 and box.conf > CONF_MINIMA_YOLO:
                coords = box.xyxy[0].tolist()
                detections_for_sort.append(coords + [box.conf.item(), box.cls.item()])

        # --- CORREÇÃO AQUI: Passando 'image' como segundo argumento ---
        tracked_objects = tracker.update(np.array(detections_for_sort), image)

        for obj in tracked_objects:
            box_x1, box_y1, box_x2, box_y2, track_id, _, _ = map(int, obj)

            roi = image[box_y1:box_y2, box_x1:box_x2]
            if roi.size == 0: 
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(roi_rgb)

            if results_pose.pose_landmarks:
                for landmark in results_pose.pose_landmarks.landmark:
                    landmark.x = (landmark.x * (box_x2 - box_x1) + box_x1) / frame_width
                    landmark.y = (landmark.y * (box_y2 - box_y1) + box_y1) / frame_height

                mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.putText(image, f"Pessoa #{track_id}", (box_x1, box_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(image)

# ------------------- FINALIZAÇÃO ---------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("-" * 30)
print(f">>> Processamento concluído!")
print(f">>> Vídeo salvo em: '{output_video_path}'")
print("-" * 30)