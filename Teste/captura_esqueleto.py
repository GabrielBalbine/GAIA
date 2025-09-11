# captura_esqueleto.py (Versão Completa e Final)

import cv2
import mediapipe as mp
import sys
import os

# ------------------- INICIALIZAÇÃO -------------------
# Inicializa as soluções do MediaPipe que vamos usar
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------- CONFIGURAÇÃO --------------------
# Coloque aqui o nome do vídeo que você vai analisar
input_video_path = "video_teste.mp4" 
output_video_path = "esqueleto_output.mp4"

# ---------------- VERIFICAÇÃO INICIAL ----------------
# Checa se o arquivo de vídeo realmente existe antes de começar
if not os.path.exists(input_video_path):
    print(f"ERRO: Arquivo de vídeo não encontrado no caminho: '{input_video_path}'")
    sys.exit(1) # Encerra o script com um código de erro

# ----------------- ABERTURA DO VÍDEO -----------------
# Abre o arquivo de vídeo de entrada
cap = cv2.VideoCapture(input_video_path)

# Verifica se o vídeo foi aberto com sucesso
if not cap.isOpened():
    print(f"ERRO: Não foi possível abrir o arquivo de vídeo. Verifique se o arquivo está corrompido ou em um formato incompatível.")
    sys.exit(1)

# Pega as informações do vídeo (largura, altura, FPS) para criar o vídeo de saída
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f">>> Vídeo aberto com sucesso! Resolução: {frame_width}x{frame_height}, FPS: {fps}")

# Define o codec e cria o objeto para escrever o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ---------------- LOOP DE PROCESSAMENTO ----------------
print(f">>> Processando vídeo '{input_video_path}'...")
frame_count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(">>> Chegou ao final do vídeo.")
            break # Encerra o loop se não houver mais frames

        frame_count += 1
        if frame_count % 30 == 0: # Imprime o progresso a cada 30 frames
            print(f"Processando frame #{frame_count}...")
            
        # Converte as cores (BGR para RGB) para o MediaPipe processar
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem para detectar a pose
        results = pose.process(image_rgb)

        # Desenha o esqueleto na imagem original (BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, # Desenha na imagem original 'image'
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Escreve o frame com o esqueleto no arquivo de vídeo de saída
        out.write(image)

# ------------------- FINALIZAÇÃO ---------------------
# Libera os objetos de vídeo e fecha tudo
cap.release()
out.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f">>> Processamento concluído!")
print(f">>> Total de {frame_count} frames processados.")
print(f">>> Vídeo com esqueleto salvo em: '{output_video_path}'")
print("-" * 30)