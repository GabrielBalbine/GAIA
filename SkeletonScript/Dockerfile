# Dockerfile (versão com correção da libGL)

FROM python:3.9

# --- CORREÇÃO AQUI: Usando o novo nome do pacote 'libgl1' ---
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sleep", "infinity"]