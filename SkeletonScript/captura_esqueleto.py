# captura_esqueleto.py (Versão final adaptada)

import cv2
import mediapipe as mp
import sys
import os

# ------------------- INICIALIZAÇÃO -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------- CONFIGURAÇÃO --------------------
input_video_path = "video_teste_2_pessoas.mp4" 
output_video_path = "esqueleto_output_final.mp4"

# ---------------- VERIFICAÇÃO INICIAL ----------------
if not os.path.exists(input_video_path):
    print(f"ERRO: Arquivo de vídeo não encontrado: '{input_video_path}'")
    sys.exit(1)

# ----------------- ABERTURA DO VÍDEO -----------------
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"ERRO: Não foi possível abrir o arquivo de vídeo.")
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

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(">>> Chegou ao final do vídeo.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processando frame #{frame_count}...")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # --- LÓGICA DE DESENHO SIMPLIFICADA ---
        # Se QUALQUER landmark foi detectado, desenhe-o
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
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
print(f">>> Vídeo com esqueletos salvo em: '{output_video_path}'")
print("-" * 30)