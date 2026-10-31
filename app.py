"""
FastAPI HTTP-сервис для Grand Mobile Support Copilot.
Реализует контракт POST /triage для хакатона Decentrathon 6.0.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, Dict
import uvicorn

from service import triage_ticket, _is_complaint_intent

app = FastAPI(
    title="Grand Mobile Support Copilot",
    description="Сервис триажа обращений игроков и подготовки черновиков ответов",
    version="1.0.0"
)

# Монтируем директорию со статикой
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    """Веб-интерфейс для демонстрации куратору и жюри"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return "<h1>Grand Mobile Support Copilot</h1><p>index.html не найден</p>"


class MessageItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str = "player"
    text: str
    attachments: Optional[int] = 0
    ts: Optional[str] = None


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Optional[str] = Field(default=None, description="Тип обращения: 'report', 'complaint' или 'question'")
    messages: Optional[List[MessageItem]] = Field(default=None, description="Сообщения в тикете")
    text: Optional[str] = Field(default=None, description="Одиночный текст тикета")
    ticket_id: Optional[str] = None
    channel: Optional[str] = None
    author: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "support-copilot"}


@app.post("/triage")
def triage_endpoint(req: TriageRequest) -> Dict[str, Any]:
    """
    Основной метод контракта POST /triage:
    Принимает JSON с тикетом, возвращает классификацию,
    извлечённые факты, черновик ответа со ссылками на правила и флаг needs_human.
    """
    try:
        payload = req.model_dump()
        raw_text = payload.get("text") or ""
        
        # Если тикет передан как простой текст (из веб-формы или curl), оборачиваем в messages
        if not payload.get("messages") and raw_text:
            payload["messages"] = [
                {
                    "role": "player",
                    "text": raw_text,
                    "attachments": 0
                }
            ]
        elif payload.get("messages") and not raw_text:
            raw_text = " ".join(m.get("text", "") for m in payload["messages"])

        # Умное определение kind, если не задан явно:
        if not payload.get("kind"):
            payload["kind"] = "complaint" if _is_complaint_intent(raw_text.lower()) else "report"

        result = triage_ticket(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
