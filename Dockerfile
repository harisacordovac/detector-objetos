FROM python:3.12-slim

# Dependencias que OpenCV necesita
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV YOLO_MODEL=yolov8n.pt
ENV HF_HOME=/app/.cache

# En Spaces el puerto por defecto es 7860. Usa $PORT si te lo pasan.
EXPOSE 7860
CMD ["sh","-c","gunicorn -w 1 -k gthread -b 0.0.0.0:${PORT:-7860} app:app --timeout 180"]
