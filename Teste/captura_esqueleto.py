import cv2
import mediapipe as mp

# Inicializa as soluções do MediaPipe que vamos usar
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Abre a câmera do seu computador (webcam).
# O Codespaces vai pedir permissão e encaminhar a imagem para o ambiente.
cap = cv2.VideoCapture(0)

# Inicializa o detector de pose do MediaPipe
# O 'with' garante que os recursos serão liberados no final
with mp_pose.Pose(
    min_detection_confidence=0.5,  # Confiança mínima para detectar uma pessoa
    min_tracking_confidence=0.5    # Confiança mínima para rastrear a pessoa
) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frame vazio da câmera.")
            continue

        image.flags.setflags(write=False)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image)

        image.flags.setflags(write=True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS, # Desenha as linhas de conexão entre os pontos
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        cv2.imshow('MediaPipe Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()