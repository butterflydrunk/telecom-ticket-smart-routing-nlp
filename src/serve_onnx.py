"""
Высокопроизводительный микросервис классификации тикетов на базе ONNX Runtime.
Работает на CPU без PyTorch/GPU зависимостей во время инференса.
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dataset import TextTokenizer, LABELS_MAP, ROUTING_DEPARTMENT_MAP

app = FastAPI(
    title="Служба 071 | Smart Ticket NLP Router",
    description="Автоматическая маршрутизация и приоритизация обращений абонентов на базе ONNX Runtime INT8",
    version="1.0.0"
)

SESSION = None
TOKENIZER = None


class TicketRequest(BaseModel):
    ticket_text: str = Field(..., min_length=5, description="Текст обращения абонента")
    subscriber_id: str = Field(default="SUB-UNKNOWN", description="Идентификатор лицевого счета")


class TicketClassificationResponse(BaseModel):
    category: str
    category_ru: str
    confidence: float
    target_department: str
    sla_priority: str
    action_note: str


@app.on_event("startup")
def init_onnx_runtime():
    global SESSION, TOKENIZER
    import onnxruntime as ort

    int8_path = os.getenv("ONNX_MODEL_PATH", "artifacts/model_int8.onnx")
    base_onnx_path = "artifacts/model.onnx"
    tok_path = os.getenv("TOKENIZER_PATH", "artifacts/tokenizer.json")

    chosen_model = int8_path if os.path.exists(int8_path) else base_onnx_path

    if os.path.exists(chosen_model) and os.path.exists(tok_path):
        TOKENIZER = TextTokenizer.load(tok_path)
        SESSION = ort.InferenceSession(chosen_model, providers=["CPUExecutionProvider"])
        print(f"[OK] ONNX Runtime сессия запущена ({chosen_model}) на CPUExecutionProvider.")
    else:
        print("[WARN] Модель ONNX не найдена. Сначала запустите src/train_and_export_onnx.py")


@app.get("/health", tags=["Infra"])
def health():
    return {"status": "healthy" if SESSION is not None else "degraded"}


@app.post("/classify-ticket", response_model=TicketClassificationResponse, tags=["NLP Routing"])
def classify_ticket(payload: TicketRequest):
    if SESSION is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="ONNX сессия не инициализирована.")

    encoded = TOKENIZER.encode(payload.ticket_text)
    input_tensor = np.array([encoded], dtype=np.int64)

    # Инференс в ONNX Runtime
    outputs = SESSION.run(None, {"input_ids": input_tensor})
    logits = outputs[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    category_code = LABELS_MAP.get(pred_idx, "UNKNOWN")
    department = ROUTING_DEPARTMENT_MAP.get(category_code, "Первая линия поддержки")

    # Приоритезация SLA
    priority_rules = {
        "TECH_INCIDENT": ("HIGH", "Аварийный регламент: время реакции до 15 минут"),
        "BILLING_DISPUTE": ("MEDIUM", "Финансовый регламент: время реакции до 1 часа"),
        "ROAMING_SERVICE": ("HIGH", "Клиент в роуминге: оперативная проверка статуса линии"),
        "TARIFF_MANAGEMENT": ("LOW", "Стандартная консультация: до 4 часов")
    }
    priority, note = priority_rules.get(category_code, ("MEDIUM", "Стандартная обработка"))

    ru_names = {
        "TECH_INCIDENT": "Технический инцидент / Сбой связи",
        "BILLING_DISPUTE": "Биллинг / Финансовый спор",
        "ROAMING_SERVICE": "Международный роуминг",
        "TARIFF_MANAGEMENT": "Управление тарифами и услугами"
    }

    return TicketClassificationResponse(
        category=category_code,
        category_ru=ru_names.get(category_code, category_code),
        confidence=round(confidence, 4),
        target_department=department,
        sla_priority=priority,
        action_note=note
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
