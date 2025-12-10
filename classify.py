"""
classify.py — tiny parser + rule-based classifier with clear comments.

Why rule-based first?
- It's deterministic, reproducible, and has zero external dependencies.
- It forms a strong "baseline" so you can measure the benefit of adding an LLM later.
"""

import re
from typing import Dict, List

# ------------------------------------------------------------------
# 1) A simple regex to extract fields from a log line
#    Example input:
#    "[ERROR] gNB-23: Packet loss increased 80% (uplink)"
# ------------------------------------------------------------------

LOG_RE = re.compile(r"""
    \[(?P<level>INFO|WARNING|ERROR)\]   # e.g., [ERROR]
    \s*
    (?P<node>gNB-\d+):                 # e.g., gNB-23:
    \s*
    (?P<message>.+)                    # the rest of the line
""", re.VERBOSE)

def parse_log_line(line: str) -> Dict:
    """
    Convert a raw log line into a small dictionary of fields.
    If the line does not match the pattern, we return best-effort defaults.
    """
    m = LOG_RE.search(line)
    if not m:
        # No match? Return a generic structure so the rest of the pipeline still works.
        return {
            "level": "INFO",
            "node": "gNB-0",
            "message": line.strip(),
            "message_lower": line.strip().lower()
        }

    d = m.groupdict()
    d["message"] = d["message"].strip()
    d["message_lower"] = d["message"].lower()
    return d

# ------------------------------------------------------------------
# 2) A tiny rule-based classifier
#    We define 5 fault classes with a few domain-inspired keywords each.
#    The algorithm scores a class by counting keyword matches in the text.
# ------------------------------------------------------------------

KEYWORDS = {
    # "congestion" is about over-utilization or dropped packets
    "congestion": [
        "packet loss", "throughput drop", "buffer overflow", "prb utilization", "congestion"
    ],
    # "backhaul" relates to transport issues between RAN and core (fiber, L2, etc.)
    "backhaul": [
        "backhaul", "uplink delay", "fiber", "l2 retransmission", "l2 retransmissions", "latency spikes"
    ],
    # "hardware" is physical component problems at RU/DU/CU (fans, optics, overheating)
    "hardware": [
        "ru failure", "overheating", "fan", "sfp", "psu", "temperature"
    ],
    # "interference" is radio issues (RSRP/SINR/PCI)
    "interference": [
        "sinr", "rsrp", "pci", "neighbor", "collision", "interference"
    ],
    # "core" points to delays/errors in UPF/SMF/AMF or core interfaces
    "core": [
        "upf", "smf", "amf", "n3 interface", "core latency", "session setup failure"
    ],
}

def score_label(text: str, label: str) -> List[str]:
    """
    Count how many keywords for a given label appear in 'text'.
    Return the list of matched keywords (empty if none).
    """
    hits = [k for k in KEYWORDS[label] if k in text]
    return hits

def predict_fault(text: str) -> Dict:
    """
    Rule-based prediction:
    - Iterate over labels, count keyword matches.
    - Pick the label with the highest count.
    - Convert that count into a simple "confidence" heuristic.
    """
    best_label = "congestion"   # fallback
    best_hits: List[str] = []
    best_count = 0

    for label in KEYWORDS:
        hits = score_label(text, label)
        if len(hits) > best_count:
            best_label = label
            best_count = len(hits)
            best_hits = hits

    # Confidence heuristic: more keyword hits => higher confidence (cap at 0.95)
    confidence = min(0.95, 0.55 + 0.1 * best_count)

    return {
        "label": best_label,
        "confidence": round(confidence, 3),
        "rationale": best_hits
    }

# ------------------------------------------------------------------
# 3) OPTIONAL: How to integrate an LLM later
#    - Call your provider's API with a prompt that includes the message text.
#    - Ask the model to choose exactly one label and provide a short rationale.
#    - Keep the same return structure (label, confidence, rationale) so your API stays stable.
# ------------------------------------------------------------------

# Example skeleton (not executed here to keep the project dependency-free):
#
# def predict_with_llm(text: str) -> Dict:
#     prompt = f\"\"\"
#     You are a telecom reliability engineer. Classify the fault in this log into one of:
#     [congestion, backhaul, hardware, interference, core].
#     Log: "{text}"
#     Return JSON: {{"label": <one of 5>, "confidence": 0..1, "rationale": [short phrases]}}
#     \"\"\"
#     # call your LLM client here and parse the JSON
#     # fallback to predict_fault(text) if the API fails
#     return predict_fault(text)
