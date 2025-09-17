# captura_esqueleto.py (Versão com Visualização de Bounding Box)

import cv2
import mediapipe as mp
import sys
import os

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print(">>> Carregando modelo de detecção de pessoas (SSD)...")
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet.pbtxt')

# --- MUDANÇA 1: Deixamos o agente menos exigente ---
ID_CLASSE_PESSOA = 15
CONF_MINIMA = 0.40 # Baixamos de 0.60 para 0.40

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_debug.mp4" # Novo nome para o vídeo de saída

# ---------------- VERIFICAÇÃO E ABERTURA DO VÍDEO ----------------
# (O resto dessa parte continua igual)
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
        
        # ETAPA 1: Detectar todas as PESSOAS
        blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # ETAPA 2: Para cada pessoa encontrada, rodar a estimação de pose
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if class_id == ID_CLASSE_PESSOA and confidence > CONF_MINIMA:
                # Pega as coordenadas do retângulo (bounding box) da pessoa
                box_x1 = int(detections[0, 0, i, 3] * frame_width)
                box_y1 = int(detections[0, 0, i, 4] * frame_height)
                box_x2 = int(detections[0, 0, i, 5] * frame_width)
                box_y2 = int(detections[0, 0, i, 6] * frame_height)

                # --- MUDANÇA 2: Desenhamos o retângulo e a confiança ---
                # Isso nos mostra o que o detector SSD está vendo
                cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2) # Retângulo amarelo
                label = f"Pessoa: {confidence:.2%}"
                cv2.putText(image, label, (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Recorta a imagem da pessoa
                roi = image[box_y1:box_y2, box_x1:box_x2]
                if roi.size == 0: continue

                # Roda o MediaPipe Pose SÓ no recorte
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results = pose.process(roi_rgb)
                
                # Desenha o esqueleto se encontrado
                if results.pose_landmarks:
                    # Precisamos converter os landmarks de volta para as coordenadas da imagem original
                    # Esta lógica precisa ser ajustada, vamos desenhar direto no ROI por enquanto para simplificar o debug
                    mp_drawing.draw_landmarks(roi, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        out.write(image)

# ------------------- FINALIZAÇÃO ---------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f">>> Processamento concluído!")
print(f">>> Vídeo de DEBUG salvo em: '{output_video_path}'")
print("-" * 30)