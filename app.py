"""
app.py — a tiny, commented FastAPI app exposing POST /diagnose

The endpoint accepts a list of raw log lines and returns, for each line:
- predicted fault class (one of 5 categories)
- confidence score (heuristic for the simple baseline)
- short rationale (which keywords triggered the decision)

This is intentionally simple and readable to help you understand and extend it.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict

from classify import parse_log_line, predict_fault

import csv
import time
from pathlib import Path

# Path object points to a CSV file in the current working directory.
LOG_PATH = Path("predictions_log.csv")

# Create the file and write a header row if it doesn't exist yet.
if not LOG_PATH.exists():
    with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["ts", "level", "node", "input", "predicted", "confidence", "rationale"]
        )



# Create the FastAPI application
app = FastAPI(
    title="Minimal Fault Diagnosis API",
    version="0.1.0",
    description="A small, well-commented baseline for 5G/6G fault diagnosis."
)


class DiagnoseRequest(BaseModel):
    # We accept a list so you can send multiple log lines in one request
    logs: List[str] = Field(..., description="Raw log lines, e.g., '[ERROR] gNB-23: Packet loss increased 80% (uplink)'")

class DiagnoseResponse(BaseModel):
    results: List[Dict] = Field(..., description="List of per-line predictions with metadata")

@app.get("/health")
def health():
    """
    Lightweight health check to confirm the service is running.
    """
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    outputs = []
    rows = []

    for raw in req.logs:
        parsed = parse_log_line(raw)
        pred = predict_fault(parsed["message_lower"])
        result = {
            "input": raw,
            "predicted": pred["label"],
            "confidence": pred["confidence"],
            "rationale": pred["rationale"],
            "meta": {"level": parsed.get("level"), "node": parsed.get("node")},
        }
        outputs.append(result)
        rows.append([
            int(time.time()),
            result["meta"]["level"],
            result["meta"]["node"],
            result["input"],
            result["predicted"],
            result["confidence"],
            "|".join(result["rationale"]),
        ])

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return DiagnoseResponse(results=outputs)

# app.py
@app.get("/")
def index():
    return {
        "message": "Fault Diagnosis API is running.",
        "endpoints": ["/docs", "/health", "/diagnose"]
    }