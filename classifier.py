# classifier.py
import re
import json
import time
from typing import Dict, Any

# Use the updated Ollama integration
from langchain_ollama import OllamaLLM

# initialize the local model (change model name if you pulled a different one)
llm = OllamaLLM(model="llama3.1", temperature=0.1, max_tokens=512)

# --- Prompt template (strict JSON-only output) ---
PROMPT_TEMPLATE = """
You are an MNPI (Material Non‑Public Information) detection classifier.
Read the following text CHUNK and decide whether it contains MNPI according to the strict definition below.

DEFINITION (strict):
- MNPI is non-public information that a reasonable investor would consider likely to affect the price or value of a company's securities, or that would be material to a decision to buy, sell, or hold securities.
- Examples: unreleased earnings/financial forecasts, confirmed but unannounced M&A, non-public regulatory investigation details, planned executive departures or appointments that would materially affect company prospects, confidential product launch timelines tied to revenue, or other confidential strategic plans with measurable financial impact.

INSTRUCTIONS (must follow exactly):
1. Do NOT quote or reproduce any verbatim text from the chunk. Never output raw text from the document.
2. Consider "public vs non-public" signals. Phrases like "announced" or "press release" reduce materiality; "confidential" or "internal" increase it.
3. Consider materiality: financial figures tied to future performance, firm dates for product launches that affect revenue, confirmed negotiations (M&A), or confirmed executive actions are strong signals.
4. Consider context: hypotheticals, examples, or generated/sample text should NOT be treated as MNPI unless it clearly states real, non-public facts.
5. Output JSON ONLY (no explanation, no markdown, no commentary). If you cannot determine, return mnpi:"unclear" with low confidence.

JSON OUTPUT SCHEMA (strict):
{
  "mnpi": "yes" | "no" | "unclear",
  "categories": ["Unreleased Earnings/Guidance", "M&A/Transactions", "Executive Changes",
                 "Product Launch/Strategic Plans", "Legal/Regulatory Investigations",
                 "Insider Trading Risk", "Confidential Financial Projections", "None"],
  "confidence": 0.00-1.00,
  "evidence_summary": "high-level one-sentence reason (no quotes, <=30 words)",
  "risk_level": "low" | "medium" | "high",
  "recommended_action": "escalate" | "human_review" | "no_action",
  "notes": "optional short note for human reviewer (max 40 words)"
}

Now analyze the chunk below.

--- CHUNK START ---
{CHUNK}
--- CHUNK END ---
"""

# --- Safe JSON extraction + normalization (same logic from the prompt guidance) ---
def safe_parse_model_output(response_text: str) -> Dict[str, Any]:
    # Extract first JSON-looking substring to be robust
    m = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if not m:
        return {
            "mnpi": "unclear",
            "categories": ["None"],
            "confidence": 0.0,
            "evidence_summary": "No valid JSON returned",
            "risk_level": "low",
            "recommended_action": "human_review",
            "notes": "Model did not return JSON"
        }
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {
            "mnpi": "unclear",
            "categories": ["None"],
            "confidence": 0.0,
            "evidence_summary": "Malformed JSON",
            "risk_level": "low",
            "recommended_action": "human_review",
            "notes": "JSON parse error"
        }

    # normalize fields
    if data.get("mnpi") not in ("yes", "no", "unclear"):
        data["mnpi"] = "unclear"

    if not isinstance(data.get("categories"), list):
        data["categories"] = ["None"]

    if not isinstance(data.get("confidence"), (int, float)):
        # support confidence as string too
        try:
            data["confidence"] = float(data.get("confidence", 0.0))
        except Exception:
            data["confidence"] = 0.0

    # clamp confidence to [0.00, 0.95] and round
    data["confidence"] = round(max(0.0, min(0.95, float(data["confidence"]))), 2)

    # apply overrides: if model says yes but not confident -> unclear
    if data["mnpi"] == "yes" and data["confidence"] < 0.5:
        data["mnpi"] = "unclear"
        data["recommended_action"] = "human_review"

    # validate categories values
    allowed = {"Unreleased Earnings/Guidance", "M&A/Transactions", "Executive Changes",
               "Product Launch/Strategic Plans", "Legal/Regulatory Investigations",
               "Insider Trading Risk", "Confidential Financial Projections", "None"}
    data["categories"] = [c for c in data["categories"] if c in allowed]
    if not data["categories"]:
        data["categories"] = ["None"]

    # risk_level sanity
    if data.get("risk_level") not in ("low", "medium", "high"):
        c = data["confidence"]
        if c >= 0.75:
            data["risk_level"] = "high"
        elif c >= 0.5:
            data["risk_level"] = "medium"
        else:
            data["risk_level"] = "low"

    # recommended_action sanity
    if data.get("recommended_action") not in ("escalate", "human_review", "no_action"):
        if data["risk_level"] == "high":
            data["recommended_action"] = "escalate"
        elif data["mnpi"] == "unclear" or data["risk_level"] == "medium":
            data["recommended_action"] = "human_review"
        else:
            data["recommended_action"] = "no_action"

    return data

# --- classify_chunk with retries and lightweight verifier ---
def classify_chunk(chunk: str, max_retries: int = 2, retry_delay: float = 0.6) -> Dict[str, Any]:
    """
    Returns a normalized dict with keys:
    mnpi, categories, confidence, evidence_summary, risk_level, recommended_action, notes
    """
    prompt = PROMPT_TEMPLATE.replace("{CHUNK}", chunk)

    last_response_text = ""
    for attempt in range(1, max_retries + 1):
        try:
            # OllamaLLM.invoke returns a string
            response_text = llm.invoke(prompt)
            last_response_text = response_text
            parsed = safe_parse_model_output(response_text)
            # If parsed and confidence > 0.75 or mnpi == "no", accept immediately
            if parsed["confidence"] >= 0.75 or parsed["mnpi"] == "no":
                return parsed
            # If parsed but low confidence, continue to verifier or retry
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            else:
                # proceed to verifier step below
                break
        except Exception as e:
            # transient error: retry
            last_response_text = f"ERROR: {e}"
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            else:
                return {
                    "mnpi": "unclear",
                    "categories": ["None"],
                    "confidence": 0.0,
                    "evidence_summary": "LLM error",
                    "risk_level": "low",
                    "recommended_action": "human_review",
                    "notes": f"LLM error after retries: {e}"
                }

    # --- Verifier pass for borderline cases ---
    parsed = safe_parse_model_output(last_response_text)
    # If borderline (confidence between 0.4 and 0.75) do a short re-check prompt
    if 0.4 <= parsed["confidence"] < 0.75:
        verifier_prompt = (
            "RE-CHECK: Based on the previous analysis, do you CONFIRM the MNPI judgement and "
            "confidence? Provide JSON with same schema (mnpi, categories, confidence, evidence_summary, "
            "risk_level, recommended_action, notes). Respond JSON only. If unsure, return mnpi:'unclear'.\n\n"
            "--- ORIGINAL CHUNK ---\n" + chunk
        )
        try:
            verifier_text = llm.invoke(verifier_prompt)
            verifier_parsed = safe_parse_model_output(verifier_text)
            # if verifier confidence is lower than original, prefer lower one and set to unclear if needed
            final_conf = min(parsed["confidence"], verifier_parsed.get("confidence", 0.0))
            # choose the safer mnpi label (prefer 'unclear' over 'yes')
            final_mnpi = parsed["mnpi"]
            if verifier_parsed["mnpi"] == "unclear" or parsed["mnpi"] == "unclear":
                final_mnpi = "unclear"
            elif verifier_parsed["mnpi"] == "no" and parsed["mnpi"] == "yes":
                # prefer unclear if conflict
                final_mnpi = "unclear"
            # build final record (take categories from verifier if it is consistent)
            final_categories = verifier_parsed["categories"] if verifier_parsed.get("categories") else parsed["categories"]
            final = {
                "mnpi": final_mnpi,
                "categories": final_categories,
                "confidence": round(max(0.0, min(0.95, float(final_conf))), 2),
                "evidence_summary": verifier_parsed.get("evidence_summary") or parsed.get("evidence_summary"),
                "risk_level": verifier_parsed.get("risk_level") or parsed.get("risk_level"),
                "recommended_action": verifier_parsed.get("recommended_action") or parsed.get("recommended_action"),
                "notes": verifier_parsed.get("notes") or parsed.get("notes")
            }
            # post-processing overrides
            if final["mnpi"] == "yes" and final["confidence"] < 0.5:
                final["mnpi"] = "unclear"
                final["recommended_action"] = "human_review"
            return final
        except Exception:
            # If verifier fails, fall back to parsed but mark for human review
            parsed["recommended_action"] = "human_review"
            parsed["notes"] = (parsed.get("notes") or "") + " | verifier failed"
            return parsed

    # Not borderline or verifier not required -> return parsed
    return parsed
