FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV YOLO_MODEL=yolov8n.pt
ENV PORT=8000
CMD ["gunicorn","-w","1","-k","gthread","-b","0.0.0.0:8000","app:app","--timeout","180"]
