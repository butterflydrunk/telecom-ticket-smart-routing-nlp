FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Установка зависимостей без тяжелого PyTorch для рантайма
RUN pip install --no-cache-dir onnxruntime==1.17.0 fastapi==0.110.0 uvicorn==0.28.0 pydantic==2.6.0 numpy==1.26.4

COPY src/ ./src/
COPY artifacts/ ./artifacts/

ENV PYTHONUNBUFFERED=1
EXPOSE 8002

CMD ["uvicorn", "src.serve_onnx:app", "--host", "0.0.0.0", "--port", "8002"]
