# analise.py
import os
import pandas as pd
from fer import FER, Video

video_path = 'video_analyse.mp4' 

video = Video(video_file=video_path)
emotions_detector = FER(mtcnn=True)

print(">>> Iniciando a análise do vídeo... pode demorar um pouco.")
data = video.analyze(detector=emotions_detector)
print(">>> Análise concluída!")

output_path = 'resultados/resultado_analise.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)

print(f">>> Resultado salvo em '{output_path}'")