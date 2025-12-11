# aggregator.py

from typing import List, Dict, Any

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalizes classifier outputs and produces an overall summary.
    Expects each result to contain keys like:
      - mnpi: "yes" | "no" | "unclear"
      - categories: list of strings (may be missing)
      - confidence: float (may be missing)
    """
    # Defensive normalization
    normalized = []
    for r in results:
        if not isinstance(r, dict):
            continue
        normalized.append({
            "mnpi": r.get("mnpi", "unclear"),
            "categories": r.get("categories", ["None"]) if isinstance(r.get("categories", None), list) else ["None"],
            "confidence": float(r.get("confidence", 0.0)) if r.get("confidence") is not None else 0.0,
            "evidence_summary": r.get("evidence_summary", ""),
            "recommended_action": r.get("recommended_action", "human_review")
        })

    mnpi_flags = [r for r in normalized if r["mnpi"] == "yes"]

    if not mnpi_flags:
        # no strong MNPI detected
        # optionally check if any 'unclear' exists and recommend human review
        any_unclear = any(r["mnpi"] == "unclear" for r in normalized)
        return {
            "overall_mnpi": "no",
            "categories": [],
            "overall_confidence": round(max((r["confidence"] for r in normalized), default=0.0), 2),
            "reason": "No MNPI detected." if not any_unclear else "No MNPI detected; some chunks marked unclear - human review recommended.",
            "recommended_action": "human_review" if any_unclear else "no_action"
        }

    # Aggregate categories from all flagged chunks
    categories_set = set()
    for r in mnpi_flags:
        for c in r.get("categories", ["None"]):
            categories_set.add(c)

    # sanitize categories (if only "None", return empty)
    if categories_set == {"None"}:
        categories = []
    else:
        categories = sorted([c for c in categories_set if c != "None"])

    # overall confidence: choose max confidence among flagged chunks
    overall_confidence = round(max(r.get("confidence", 0.0) for r in mnpi_flags), 2)

    return {
        "overall_mnpi": "yes",
        "categories": categories,
        "overall_confidence": overall_confidence,
        "reason": f"Detected MNPI in {len(mnpi_flags)} chunk(s).",
        "recommended_action": "escalate" if overall_confidence >= 0.75 else "human_review"
    }
