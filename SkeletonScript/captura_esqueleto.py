# captura_esqueleto.py (A Solução Definitiva com YOLOv8-Pose)

import cv2
from ultralytics import YOLO
import sys
import os
import numpy as np

# ------------------- INICIALIZAÇÃO -------------------
print(">>> Carregando modelo de Pose (YOLOv8s-pose)...")
# O modelo de pose será baixado automaticamente na primeira execução
model = YOLO('yolov8s-pose.pt') 

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_YOLOPose.mp4"
CONF_MINIMA = 0.50

# Dicionário para manter uma cor única para cada pessoa rastreada
cores_pessoas = {}
# Função para gerar uma cor aleatória
def cor_aleatoria(track_id):
    if track_id not in cores_pessoas:
        # Gera uma cor vibrante e fácil de ver
        cores_pessoas[track_id] = tuple(np.random.randint(100, 256, size=3).tolist())
    return cores_pessoas[track_id]

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

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print(">>> Fim do vídeo.")
        break
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processando frame #{frame_count}...")
    
    # Roda o modelo de tracking e pose do YOLO
    # 'persist=True' diz ao tracker para lembrar das pessoas entre os frames
    # classes=[0] foca a detecção apenas em 'pessoa'
    results = model.track(image, persist=True, verbose=False, classes=[0])

    if results[0].boxes.id is not None:
        # Pega os keypoints (pontos do esqueleto) e os IDs de rastreamento
        keypoints = results[0].keypoints.xy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        # Itera sobre cada pessoa rastreada
        for kpts, track_id in zip(keypoints, track_ids):
            cor = cor_aleatoria(track_id)
            
            # Desenha as conexões do esqueleto (linhas)
            conexoes = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Cabeça
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), # Tronco
                (11, 13), (13, 15), (12, 14), (14, 16) # Pernas
            ]
            for i, j in conexoes:
                # Verifica se os pontos foram detectados antes de desenhar a linha
                if i < len(kpts) and j < len(kpts) and kpts[i][0] > 0 and kpts[j][0] > 0:
                    cv2.line(image, tuple(kpts[i]), tuple(kpts[j]), cor, 2)
            
            # Desenha os pontos (juntas)
            for kpt in kpts:
                if kpt[0] > 0: # Só desenha se o ponto foi detectado
                    cv2.circle(image, tuple(kpt), 4, cor, -1)

            # Desenha o ID da pessoa
            bbox = results[0].boxes.xyxy.cpu().numpy()[list(track_ids).index(track_id)].astype(int)
            cv2.putText(image, f"ID: {track_id}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    
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
