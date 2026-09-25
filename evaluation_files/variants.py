"""Stage 6 — Three system variants for the comparison required in bullet 2.

A) Minimal LLM        — no retrieval, no rubric, no guardrail. Own knowledge only.
B) Simplified system  — real retrieval, but the LLM assigns the verdict itself
                         (no SFA rubric) and there is no guardrail on the output.
C) Full system        — chain.analyse_label exactly as built: rubric decides the
                         verdict, output is the deterministic explanation the app
                         actually shows to the user.

CHANGES vs the previous version (why the first evaluation was misleading):
  1. C is now scored on result["explanation"] (what app.py displays), not on
     result["llm_explanation"] (per-ingredient LLM notes). The old code judged
     text that users do not see as the main answer.
  2. Every output carries `judge_reference`: the FULL retrieved record (incl.
     ingredients and rubric reasoning) so the judge checks all variants against
     the same evidence.
  3. A and B are asked to start with "VERDICT: <ok|care|avoid|unknown>" and the
     verdict is parsed, so verdict accuracy can be computed in code.
  4. LLM/API failures set error=True so they are reported, not scored.
  5. B no longer echoes raw user input in its "not found" message.
"""

import json
import os

import requests
from dotenv import load_dotenv

import chain  # the team's real chain.py
from eval_utils import parse_verdict

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")

VERDICT_FORMAT = " Begin your answer with exactly 'VERDICT: <ok|care|avoid|unknown>' on the first line."


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    if not OPENROUTER_API_KEY:
        return "(LLM unavailable: OPENROUTER_API_KEY not set.)"
    if not user_prompt.strip():
        return "(LLM call skipped: empty input.)"  # the API rejects empty messages
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": OPENROUTER_MODEL,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"(LLM call failed: {exc})"


def _lookup_product(barcode: str):
    return chain.lookup_pinecone_barcode(barcode) or chain.lookup_openfoodfacts_barcode(barcode)


def _safe_barcode_label(barcode: str) -> str:
    """Only echo digit strings back to the user; never echo free text."""
    return barcode if barcode and barcode.isdigit() else ""


def _reference_from_record(record: dict) -> dict:
    """Full evidence the judge checks every variant against."""
    product = record["details"]
    additives = chain._extract_additives_and_highlights(product)["additives"]
    verdict, reasons = chain.apply_rubric(product.get("nutriscore_grade"), product.get("nova_group"), additives)
    return {
        "name": product.get("product_name"),
        "nutriscore_grade": product.get("nutriscore_grade"),
        "nova_group": product.get("nova_group"),
        "additives": additives,
        "ingredients_text": record.get("ingredients_text"),
        "nutrition": record.get("nutrition"),
        "rubric_verdict": verdict,
        "rubric_reasons": reasons,
    }


def _finalize(variant, verdict, explanation, grounded, guardrail_applied,
              product_data_used=None, judge_reference=None, extra=None) -> dict:
    """Attach a code-based safety check to every variant's output."""
    text = explanation or ""
    out = {
        "variant": variant,
        "verdict": verdict,
        "explanation": explanation,
        "grounded_in_real_data": grounded,
        "guardrail_applied": guardrail_applied,
        "banned_phrases_found": chain.check_output(text),
        "product_data_used": product_data_used,
        "judge_reference": judge_reference,
        "error": text.startswith("(LLM"),
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Variant A — Minimal LLM
# ---------------------------------------------------------------------------

VARIANT_A_SYSTEM = (
    "You are a food product assistant. The user will give you a barcode or a question "
    "about a food or beverage product. Give a verdict (ok, care, or avoid) and a short "
    "explanation, using your own knowledge. You have no external database access."
    + VERDICT_FORMAT
)


def variant_a_minimal_llm(input_text: str) -> dict:
    raw_output = _call_llm(VARIANT_A_SYSTEM, input_text)
    return _finalize("A_minimal_llm", verdict=parse_verdict(raw_output), explanation=raw_output,
                     grounded=False, guardrail_applied=False)


# ---------------------------------------------------------------------------
# Variant B — Simplified system (real retrieval, LLM-assigned verdict)
# ---------------------------------------------------------------------------

VARIANT_B_SYSTEM = (
    "You are a food product analyst. Using ONLY the product data given below, assign a "
    "verdict of ok, care, or avoid, and explain your reasoning in 2-3 sentences."
    + VERDICT_FORMAT
)


def variant_b_simplified(input_text: str) -> dict:
    barcode = chain.read_barcode(input_text)
    record = _lookup_product(barcode)
    if not record:
        label = _safe_barcode_label(barcode)
        message = f"Barcode {label} was not found." if label else "No valid barcode was detected in the input."
        return _finalize("B_simplified", verdict="unknown", explanation=message,
                         grounded=False, guardrail_applied=False)

    reference = _reference_from_record(record)
    product_summary = {  # what B's LLM actually sees (deliberately limited)
        "name": reference["name"],
        "nutriscore_grade": reference["nutriscore_grade"],
        "nova_group": reference["nova_group"],
        "additives": reference["additives"],
    }
    raw_output = _call_llm(VARIANT_B_SYSTEM, f"PRODUCT DATA\n{json.dumps(product_summary, ensure_ascii=False)}")
    return _finalize("B_simplified", verdict=parse_verdict(raw_output), explanation=raw_output,
                     grounded=True, guardrail_applied=False,
                     product_data_used=product_summary, judge_reference=reference)


# ---------------------------------------------------------------------------
# Variant C — Full system (rubric decides, deterministic explanation shown)
# ---------------------------------------------------------------------------

def variant_c_full_system(input_text: str) -> dict:
    result = chain.analyse_label(input_text)
    product = result.get("product") or {}
    reference = None
    if product:
        reference = {
            "name": product.get("product_name"),
            "nutriscore_grade": result.get("nutriscore_grade"),
            "nova_group": product.get("nova_group"),
            "additives": result.get("additives_highlights", {}).get("additives"),
            "ingredients_text": ", ".join(result.get("ingredients") or []),
            "nutrition": result.get("nutrition"),
            "rubric_verdict": result.get("rubric_verdict"),
            "rubric_reasons": result.get("rubric_reasons"),
        }
    llm_notes = result.get("llm_explanation")
    return _finalize(
        "C_full_system",
        verdict=result.get("rubric_verdict"),
        explanation=result.get("explanation"),          # what the app shows
        grounded=bool(result.get("external_source")),
        guardrail_applied=bool(result.get("violations")),
        product_data_used=reference,
        judge_reference=reference,
        extra={
            "llm_ingredient_notes": llm_notes,          # kept for Stage 7, not scored
            "banned_phrases_in_llm_notes": chain.check_output(llm_notes or ""),
            "rubric_reasons": result.get("rubric_reasons"),
        },
    )
