# analyzer.py  (or inside app.py where analyze_document is defined)
import json
from typing import Any
from loaders import load_document, chunk_text
from classifier import classify_chunk

def _normalize_classify_output(result: Any) -> dict:
    """
    Accepts:
      - dict (already parsed)
      - JSON string (model returned stringified JSON)
      - object with .content (some LLM wrappers)
    Returns a normalized dict.
    """
    # already a dict
    if isinstance(result, dict):
        return result

    # object with .content attribute (OpenAI ChatOpenAI style)
    if hasattr(result, "content"):
        raw = result.content
    else:
        raw = result  # probably a string

    # if bytes, decode
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    # if it's already a str, try to parse JSON substring robustly
    if isinstance(raw, str):
        # try direct json.loads first
        try:
            return json.loads(raw)
        except Exception:
            # fallback: extract first {...} block
            import re
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
    # final fallback: return an "unclear" dict so pipeline continues
    return {
        "mnpi": "unclear",
        "categories": ["None"],
        "confidence": 0.0,
        "evidence_summary": "Could not parse model output",
        "risk_level": "low",
        "recommended_action": "human_review",
        "notes": "Normalization fallback used"
    }


def analyze_document(path):
    text = load_document(path)
    chunks = chunk_text(text)

    print(f"Document loaded. {len(chunks)} chunks created.\n")

    results = []
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i+1}/{len(chunks)}...")
        raw_result = classify_chunk(chunk)   # may return dict or string or object
        normalized = _normalize_classify_output(raw_result)
        results.append(normalized)

    return results
