# Python 3.10 required -- TF 2.10 breaks on 3.11+
FROM python:3.10-slim

# System libraries OpenCV needs to open a display and read frames
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt


COPY main_final.py .
COPY Models/ ./Models/

CMD ["python", "main_final.py"]