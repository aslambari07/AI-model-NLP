from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.predict import predict_text


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

app = FastAPI(title="AI News Classifier", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class PredictRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "accuracy": metrics.get("accuracy"),
        },
    )


@app.post("/predict")
async def predict(payload: PredictRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")

    try:
        result = predict_text(text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(result)
