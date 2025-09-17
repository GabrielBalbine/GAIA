# captura_esqueleto.py (Versão Profissional com Detector SSD)

import cv2
import mediapipe as mp
import sys
import os

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Carrega o modelo de detecção de objetos (SSD com MobileNet) pré-treinado
print(">>> Carregando modelo de detecção de pessoas (SSD)...")
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet.pbtxt')
# O modelo COCO foi treinado em 90 classes, 'person' (pessoa) é a classe 15.
ID_CLASSE_PESSOA = 15
CONF_MINIMA = 0.60 # Confiança mínima para considerar uma detecção

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_ssd.mp4"

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

with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(">>> Fim do vídeo.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processando frame #{frame_count}...")
            
        # ETAPA 1: Detectar todas as PESSOAS no frame com o modelo SSD
        blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # ETAPA 2: Para cada pessoa encontrada, rodar a estimação de pose
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            # Filtra para ter certeza que é uma pessoa e que a confiança é alta
            if class_id == ID_CLASSE_PESSOA and confidence > CONF_MINIMA:
                # Pega as coordenadas do retângulo (bounding box) da pessoa
                box_x = int(detections[0, 0, i, 3] * frame_width)
                box_y = int(detections[0, 0, i, 4] * frame_height)
                box_w = int(detections[0, 0, i, 5] * frame_width)
                box_h = int(detections[0, 0, i, 6] * frame_height)

                # Recorta a imagem da pessoa
                roi = image[box_y:box_h, box_x:box_w]

                if roi.size == 0: continue

                # Roda o MediaPipe Pose SÓ no recorte
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results = pose.process(roi_rgb)
                
                # Desenha o esqueleto se encontrado
                if results.pose_landmarks:
                    # Converte os landmarks de volta para as coordenadas da imagem original
                    for landmark in results.pose_landmarks.landmark:
                        landmark.x = (landmark.x * (box_w - box_x) + box_x) / frame_width
                        landmark.y = (landmark.y * (box_h - box_y) + box_y) / frame_height
                    
                    # Desenha na imagem COMPLETA
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        out.write(image)

# ------------------- FINALIZAÇÃO ---------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f">>> Processamento concluído!")
print(f">>> Vídeo salvo em: '{output_video_path}'")
print("-" * 30)