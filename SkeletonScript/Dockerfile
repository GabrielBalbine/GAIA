# Dockerfile - Corrigido

FROM python:3.9

WORKDIR /app

# A linha abaixo foi corrigida com o novo nome do pacote (libgl1)
RUN apt-get update && apt-get install -y ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

# O resto do arquivo continua igual
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sleep", "infinity"]