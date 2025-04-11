# Usa un'immagine base con Python e PyTorch preinstallato
FROM pytorch/pytorch:latest

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia i file del progetto nella directory di lavoro
COPY . /app/

# Installa le dipendenze del progetto
RUN pip install --no-cache-dir -r requirements.txt

# Esponi la porta che il server Flower utilizzerà (se necessario)
EXPOSE 9091 9092 9093

