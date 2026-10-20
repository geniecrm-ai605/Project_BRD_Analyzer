"""FastAPI service.  Run:  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

Power Automate -> HTTP action -> POST /estimate/base64
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is cached after training: no internet calls / retries / warnings

import base64
import io
import json
import os
import time
import uuid

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from src.brd_reader import condense_docx, ensure_condensed
from src.pipeline import ROLES, StaffingModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/staffing_model.joblib")
API_KEY = os.getenv("STAFFING_API_KEY", "")  # set this in production
FB_DIR = os.getenv("FEEDBACK_DIR", "/tmp/feedback" if os.getenv("VERCEL") else "data/feedback")
try:
    os.makedirs(FB_DIR, exist_ok=True)
except OSError:
    FB_DIR = "/tmp/feedback"
    os.makedirs(FB_DIR, exist_ok=True)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CRM GENIE Staffing Recommendations")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = os.getenv("MODEL_PATH", "models/staffing_model.joblib")
        logger.info(f"Loading model from {model_path}")
        model = StaffingModel.load(model_path)
        # Warm up prediction to load embedder
        model.predict("Objective: warm up. Scope: a; b. Integrations: none. Data: no migration. Non-functional needs: none. Rollout: none.")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model")
        raise RuntimeError(f"Model loading failed: {e}")

_cache = {}


def _auth(key):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "invalid api key")


def _log(path, row):
    with open(os.path.join(FB_DIR, path), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _estimate(condensed):
    if condensed not in _cache:
        if len(_cache) > 500:
            _cache.clear()
        _cache[condensed] = model.predict(condensed)
    result = dict(_cache[condensed])
    result["request_id"] = uuid.uuid4().hex[:12]
    result["condensed_brd"] = condensed
    _log("requests.jsonl", {"request_id": result["request_id"], "ts": time.time(),
                            "condensed_brd": condensed, "prediction": _cache[condensed]})
    return result


class Base64In(BaseModel):
    file_name: str = "brd.docx"
    file_content_base64: str


class TextIn(BaseModel):
    brd_text: str


class Actual(BaseModel):
    duration_weeks: int
    business_analyst: int
    qa: int
    developers: int
    solution_architect: int
    delivery_lead: int
    infrastructure: int


class FeedbackIn(BaseModel):
    request_id: str
    project_name: str = ""
    actual: Actual


@app.get("/health")
def health():
    return {"status": "ok", "trained_on": model.n_train}


@app.post("/estimate/base64")
def estimate_base64(body: Base64In, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    try:
        raw = base64.b64decode(body.file_content_base64)
        text = condense_docx(io.BytesIO(raw)) if body.file_name.lower().endswith(".docx") \
            else ensure_condensed(raw.decode("utf-8", "ignore"))
    except Exception as e:
        raise HTTPException(400, f"could not read file: {e}")
    return _estimate(text)


@app.post("/estimate/file")
def estimate_file(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    _auth(x_api_key)
    raw = file.file.read()
    try:
        text = condense_docx(io.BytesIO(raw)) if file.filename.lower().endswith(".docx") \
            else ensure_condensed(raw.decode("utf-8", "ignore"))
    except Exception as e:
        raise HTTPException(400, f"could not read file: {e}")
    return _estimate(text)


@app.post("/estimate/text")
def estimate_text(body: TextIn, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return _estimate(ensure_condensed(body.brd_text))


@app.post("/feedback")
def feedback(body: FeedbackIn, x_api_key: str = Header(default="")):
    """Call this when the project's real team/duration is known. Used by `train.py --use-feedback`."""
    _auth(x_api_key)
    _log("actuals.jsonl", {"request_id": body.request_id, "project_name": body.project_name,
                           "actual": body.actual.model_dump(), "ts": time.time()})
    return {"saved": True}
