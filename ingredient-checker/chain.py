"""Barcode -> Pinecone product -> OpenAI ingredient health assessment."""

import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pinecone import Pinecone

from barcode_detector import extract_text

# Load .env from this file's own directory so keys resolve regardless of the caller's cwd
load_dotenv(Path(__file__).resolve().parent / ".env")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
PRODUCT_INDEX_NAME = os.getenv("PINECONE_PRODUCT_INDEX", "ingrid-beverages")
OPENFOODFACTS_API_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
VALID_VERDICTS = {"ok", "care", "avoid", "unknown"}
SFA_ADDITIVES_SOURCE = "https://www.sfa.gov.sg/docs/default-source/tools-and-resources/list-of-food-additives-permitted-under-food-regulations673e4fa37bb7440d9c23905a343796fe.pdf"
SFA_ADDITIVES_AS_OF = "31 May 2024"
# E/INS codes transcribed from the SFA permitted-additives guidance.
SFA_PERMITTED_ADDITIVES = frozenset({
    "e100", "e101", "e102", "e104", "e110", "e120", "e122", "e123", "e124", "e127", "e129", "e132", "e133",
    "e140", "e141", "e142", "e150a", "e150b", "e150c", "e150d", "e151", "e153", "e155", "e160a", "e160b",
    "e160c", "e160d", "e160e", "e161b", "e162", "e170", "e171", "e172", "e173", "e174", "e175", "e200",
    "e202", "e203", "e210", "e211", "e212", "e218", "e219", "e220", "e221", "e222", "e223", "e224", "e226",
    "e227", "e228", "e234", "e235", "e242", "e249", "e250", "e251", "e252", "e260", "e261", "e262", "e263",
    "e270", "e280", "e281", "e282", "e283", "e290", "e296", "e297", "e300", "e301", "e302", "e304", "e306",
    "e307", "e310", "e311", "e312", "e315", "e316", "e319", "e320", "e321", "e322", "e325", "e326", "e327",
    "e330", "e331", "e332", "e333", "e334", "e335", "e336", "e337", "e338", "e339", "e340", "e341", "e350",
    "e351", "e352", "e354", "e355", "e363", "e380", "e385", "e392", "e400", "e401", "e402", "e403", "e404",
    "e405", "e406", "e407", "e407a", "e410", "e412", "e413", "e414", "e415", "e416", "e417", "e418", "e420",
    "e421", "e422", "e425", "e432", "e433", "e434", "e435", "e436", "e440", "e442", "e444", "e445", "e450",
    "e451", "e452", "e459", "e460", "e461", "e462", "e463", "e464", "e465", "e466", "e468", "e469", "e470a",
    "e471", "e472a", "e472b", "e472c", "e472d", "e472e", "e473", "e475", "e476", "e477", "e481", "e482", "e483",
    "e491", "e492", "e493", "e494", "e495", "e500", "e501", "e503", "e504", "e507", "e508", "e509", "e511", "e513",
    "e514", "e515", "e516", "e517", "e522", "e524", "e525", "e526", "e527", "e528", "e529", "e530", "e535", "e536",
    "e541", "e551", "e552", "e553a", "e553b", "e554", "e556", "e575", "e576", "e577", "e578", "e579", "e585",
    "e620", "e621", "e622", "e623", "e624", "e625", "e626", "e627", "e628", "e629", "e630", "e631", "e632", "e633",
    "e634", "e635", "e640", "e641", "e650", "e900", "e901", "e902", "e903", "e904", "e905", "e920", "e941", "e942",
    "e950", "e951", "e952", "e953", "e954", "e955", "e957", "e960", "e960a", "e960b", "e960c", "e960d", "e961",
    "e964", "e965", "e966", "e967", "e968", "e969", "e999", "e1105", "e1200", "e1202", "e1204", "e1404", "e1410",
    "e1412", "e1413", "e1414", "e1420", "e1422", "e1440", "e1442", "e1450", "e1451", "e1505", "e1517", "e1518",
    "e1519", "e1520", "e1521",
})
# These permitted additives carry labeling, intake, or usage considerations in the SFA guidance.
CARE_ADDITIVES = frozenset({
    "e102", "e110", "e122", "e124", "e129", "e133", "e220", "e221", "e222", "e223", "e224", "e226", "e227", "e228",
    "e249", "e250", "e251", "e252", "e320", "e321", "e420", "e950", "e951", "e952", "e954", "e955", "e960",
})
AVOID_ADDITIVES = frozenset()
AVOID_NUTRISCORE = frozenset({"d", "e"})
CARE_NUTRISCORE = frozenset({"c"})
AVOID_NOVA_GROUP = 4
CARE_NOVA_GROUP = 3
BANNED_PHRASES = [
    "cures", "will prevent", "treats your", "safe for you",
    "you should stop eating", "diagnos", "prescrib",
]


def get_openai_client():
    """Return an OpenAI client only when the API key is configured."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1") if api_key else None


def _extract_nutrition(product_text):
    """Read the per-100g nutrition JSON preserved in the indexed product text."""
    match = re.search(
        r"Nutrition per 100g: (\[.*?\])(?:\nNutri-Score:|$)",
        product_text,
        flags=re.DOTALL,
    )
    if not match:
        return {}

    try:
        rows = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    return {
        row["name"]: {"value": row["100g"], "unit": row.get("unit", "")}
        for row in rows
        if isinstance(row, dict) and row.get("name") and row.get("100g") is not None
    }


def _extract_additives_and_highlights(metadata):
    """Derive additive codes and notable ingredient tags from product metadata."""
    tags = [tag.strip() for tag in str(metadata.get("ingredients_tags", "")).split(";") if tag.strip()]
    additives = []
    highlights = []
    for tag in tags:
        label = tag.removeprefix("en:").replace("-", " ").title()
        if re.fullmatch(r"en:e\d+[a-z]*", tag.lower()):
            additives.append(label)
        elif tag.startswith("en:"):
            highlights.append(label)
    return {
        "additives": list(dict.fromkeys(additives)),
        "highlights": list(dict.fromkeys(highlights))[:12],
    }


def _dedupe_ingredients_text(text):
    """Collapse repeated identical segments in a semicolon-joined ingredients string."""
    segments = [segment.strip() for segment in str(text or "").split(";")]
    deduped = list(dict.fromkeys(segment for segment in segments if segment))
    return "; ".join(deduped)


def lookup_pinecone_barcode(barcode):
    """Fetch a product from Pinecone using its exact barcode vector ID."""
    code = str(barcode).strip()
    if not re.fullmatch(r"\d{8,14}", code):
        return None

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        index = Pinecone(api_key=api_key).Index(PRODUCT_INDEX_NAME)
        vector = index.fetch(ids=[code]).vectors.get(code)
    except Exception:
        return None

    if not vector:
        return None

    metadata = dict(vector.metadata or {})
    metadata["ingredients_text"] = _dedupe_ingredients_text(metadata.get("ingredients_text", ""))
    return {
        "details": metadata,
        "ingredients_text": metadata["ingredients_text"],
        "nutrition": _extract_nutrition(metadata.get("text", "")),
        "source": f"Pinecone: {PRODUCT_INDEX_NAME}",
    }


def lookup_openfoodfacts_barcode(barcode):
    """Fetch a product from the OpenFoodFacts API when it is missing from Pinecone."""
    code = str(barcode).strip()
    if not re.fullmatch(r"\d{8,14}", code):
        return None

    try:
        response = requests.get(
            OPENFOODFACTS_API_URL.format(code=code),
            timeout=10,
            headers={"User-Agent": "Ingrid-Ingredient-Checker/1.0 (contact@example.com)"},
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    if payload.get("status") != 1:
        return None

    product = payload.get("product") or {}
    nutriments = product.get("nutriments") or {}
    nutrition = {}
    for key, value in nutriments.items():
        if not key.endswith("_100g") or value is None:
            continue
        name = key[: -len("_100g")]
        nutrition[name] = {"value": value, "unit": nutriments.get(f"{name}_unit", "")}

    metadata = dict(product)
    metadata["categories_tags"] = ";".join(product.get("categories_tags") or [])
    combined_tags = (product.get("ingredients_tags") or []) + (product.get("additives_tags") or [])
    metadata["ingredients_tags"] = ";".join(dict.fromkeys(combined_tags))
    ingredients_text = _dedupe_ingredients_text(
        product.get("ingredients_text") or product.get("ingredients_text_en", "")
    )
    metadata["ingredients_text"] = ingredients_text

    return {
        "details": metadata,
        "ingredients_text": ingredients_text,
        "nutrition": nutrition,
        "source": "OpenFoodFacts API",
    }


def read_barcode(source):
    """Return a barcode from a raw code or barcode-image path."""
    if source is None:
        return ""
    candidate = str(source).strip()
    if not candidate:
        return ""
    try:
        if Path(candidate).is_file():
            barcode, _ = extract_text(candidate)
            return barcode
    except OSError:
        pass
    return candidate


def parse_ingredients(ingredients_text):
    """Split product ingredient text into clean, unique labels."""
    cleaned = str(ingredients_text or "").lower()
    if "ingredients" in cleaned:
        cleaned = cleaned.split("ingredients", 1)[1].lstrip(": ")
    cleaned = cleaned.replace("(", ",").replace(")", ",")
    cleaned = re.sub(r"[;/\n]+", ",", cleaned)
    ingredients = []
    for part in cleaned.split(","):
        ingredient = re.sub(r"\s+", " ", part).strip(" .,:()")
        if re.fullmatch(r"\d+%?", ingredient):
            continue
        if not ingredient or len(ingredient) <= 1:
            continue
        if len(ingredient) > 40:
            continue
        if "@" in ingredient or "www." in ingredient or "http" in ingredient:
            continue
        if re.search(r"\blic\.?\s*no\b|\bfssai\b|\bmfd\.?\s*by\b|\bcustomer\s*care\b|\bfeedback\b", ingredient):
            continue
        if sum(ch.isdigit() for ch in ingredient) >= 5:
            continue
        ingredients.append(ingredient)
    return list(dict.fromkeys(ingredients))


def _health_prompt(product, ingredients, nutriscore_grade):
    return f"""You are a cautious food product analyst. Assess the product using ONLY the supplied product details, ingredient list, nutrition values, and Nutri-Score.

Return valid JSON with exactly these keys:
- verdicts: object mapping every listed ingredient to ok, care, avoid, or unknown
- summary: short plain-language product assessment
- ingredient_notes: object mapping each ingredient to one short reason

Rules:
- Nutri-Score is a product-level signal, not proof an ingredient is healthy or harmful.
- Use care for high sugar, salt, saturated fat, or ingredients that deserve moderation.
- Use avoid only for a clearly established serious concern; use unknown when information is insufficient.
- Do not provide medical, dietary, or treatment advice.
- Do not say a product is safe or unsafe for a person or condition.
- Every verdict key must exactly match an item in the ingredient list.

PRODUCT DETAILS
{json.dumps(product, ensure_ascii=False)}

INGREDIENTS
{json.dumps(ingredients, ensure_ascii=False)}

NUTRI-SCORE
{nutriscore_grade or "Not available"}
"""


def check_output(text):
    """Flag medical-claim language in generated output."""
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def _normalise_additive_code(code):
    """Normalize E/INS labels from product metadata for rubric matching."""
    value = str(code or "").strip().lower()
    if not value:
        return ""
    return value if value.startswith("e") else f"e{value}"


def get_flagged_additives(additive_codes):
    """Return additive codes that trigger a rubric care/avoid rule, with reasons."""
    flagged = []
    normalized_codes = list(dict.fromkeys(
        code for code in (_normalise_additive_code(value) for value in (additive_codes or [])) if code
    ))
    for code in normalized_codes:
        if code in AVOID_ADDITIVES:
            reason = "flagged for avoidance"
        elif code not in SFA_PERMITTED_ADDITIVES:
            reason = f"not found in SFA permitted-additives guidance ({SFA_ADDITIVES_AS_OF})"
        elif code in CARE_ADDITIVES:
            reason = "warrants moderation"
        else:
            continue
        flagged.append({"code": code.upper(), "reason": reason})
    return flagged


def apply_rubric(nutriscore_grade, nova_group, additive_codes):
    """Return (verdict, reasons) using deterministic SFA and nutrition rules."""
    reasons = []
    grade = str(nutriscore_grade or "").strip().lower()
    try:
        nova = int(nova_group) if nova_group is not None and str(nova_group).strip() else None
    except (TypeError, ValueError):
        nova = None

    normalized_codes = list(dict.fromkeys(
        code for code in (_normalise_additive_code(value) for value in (additive_codes or [])) if code
    ))
    avoid_hits = [code for code in normalized_codes if code in AVOID_ADDITIVES]
    unlisted_hits = [code for code in normalized_codes if code not in SFA_PERMITTED_ADDITIVES]
    care_hits = [code for code in normalized_codes if code in CARE_ADDITIVES]

    if avoid_hits:
        reasons.append(f"contains additive(s) flagged for avoidance: {', '.join(avoid_hits)}")
    if unlisted_hits:
        reasons.append(
            f"additive(s) not found in the SFA permitted-additives guidance ({SFA_ADDITIVES_AS_OF}): "
            f"{', '.join(unlisted_hits)}"
        )
    if grade in AVOID_NUTRISCORE:
        reasons.append(f"Nutri-Score {grade.upper()}")
    if nova == AVOID_NOVA_GROUP:
        reasons.append(f"NOVA group {nova}")
    if reasons:
        return "avoid", reasons

    if care_hits:
        reasons.append(f"contains additive(s) that warrant moderation: {', '.join(care_hits)}")
    if grade in CARE_NUTRISCORE:
        reasons.append(f"Nutri-Score {grade.upper()}")
    if nova == CARE_NOVA_GROUP:
        reasons.append(f"NOVA group {nova} (ultra-processed)")
    if reasons:
        return "care", reasons

    if grade in {"a", "b"}:
        return "ok", [f"Nutri-Score {grade.upper()}"]

    return "unknown", ["insufficient data to apply the rubric (no Nutri-Score or NOVA group on file)"]


def _format_number(value):
    """Format numeric nutrition values without unnecessary decimal noise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not available"
    return f"{number:.1f}" if number % 1 else f"{number:.0f}"


def build_verdict_summary(product, nutrition, rubric_verdict, rubric_reasons):
    """Build a deterministic, neutral summary from retrieved product data and the rubric."""
    name = product.get("product_name") or product.get("product_name_en") or "This product"
    title = f"# {name} – {rubric_verdict.title()}"
    grade = str(product.get("nutriscore_grade") or "").strip().upper()
    nova = product.get("nova_group")
    sugars = (nutrition.get("sugars") or {}).get("value")
    added_sugars = (nutrition.get("added_sugars") or {}).get("value")
    protein = (nutrition.get("proteins") or {}).get("value")
    fiber = (nutrition.get("fiber") or {}).get("value")
    sentence_parts = []

    if grade:
        grade_descriptions = {
            "A": "the highest possible score indicating the strongest nutritional profile",
            "B": "a relatively favorable nutritional profile",
            "C": "a middle-range nutritional profile",
            "D": "a poor nutritional profile",
            "E": "the lowest possible score indicating poor nutritional quality",
        }
        sentence_parts.append(f"This product receives a Nutri-Score {grade} rating, which is {grade_descriptions.get(grade, 'the recorded nutritional score')}.")
    elif rubric_verdict == "unknown":
        sentence_parts.append("There is not enough recorded Nutri-Score or NOVA data to fully apply the rubric.")

    nutrition_sentence = []
    if sugars is not None:
        nutrition_sentence.append(f"{_format_number(sugars)}g of sugars")
    if added_sugars is not None:
        nutrition_sentence.append(f"the product record lists {_format_number(added_sugars)}g of added sugars")
    if nutrition_sentence:
        sentence_parts.append("Per 100g, it contains " + ", and ".join(nutrition_sentence) + ".")

    low_nutrition = []
    if protein is not None and float(protein) == 0:
        low_nutrition.append("no protein")
    if fiber is not None and float(fiber) == 0:
        low_nutrition.append("no fiber")
    if low_nutrition:
        sentence_parts.append("The recorded nutrition shows " + " and ".join(low_nutrition) + ".")

    if nova is not None:
        try:
            nova_number = int(nova)
        except (TypeError, ValueError):
            nova_number = None
        if nova_number == 4:
            sentence_parts.append("As an ultra-processed beverage (NOVA Group 4), it offers limited nutritional benefit relative to its recorded sugar content.")
        elif nova_number == 3:
            sentence_parts.append("It is classified as a processed food (NOVA Group 3).")

    if rubric_reasons and not sentence_parts:
        sentence_parts.append("The rubric was applied using the available product and additive data: " + "; ".join(rubric_reasons) + ".")

    return title + "\n\n" + " ".join(sentence_parts)


def analyse_label(source, skip_llm=False):
    """Fetch barcode product details and generate LLM ingredient verdicts."""
    barcode = read_barcode(source)
    product_record = lookup_pinecone_barcode(barcode) or lookup_openfoodfacts_barcode(barcode)
    if not product_record:
        return {
            "ingredients": [], "verdicts": {}, "matches": {}, "unknowns": [],
            "product": None, "nutriscore_grade": None, "nutrition": {},
            "additives_highlights": {"additives": [], "highlights": []},
            "flagged_additives": [],
            "rubric_verdict": "unknown",
            "rubric_reasons": ["no product data available to apply the rubric"],
            "explanation": f"Barcode {barcode or 'not detected'} was not found in Pinecone or OpenFoodFacts.",
            "violations": [], "external_source": None,
        }

    product = product_record["details"]
    ingredients = parse_ingredients(product_record["ingredients_text"])
    additive_codes = _extract_additives_and_highlights(product)["additives"]
    flagged_additives = get_flagged_additives(additive_codes)
    rubric_verdict, rubric_reasons = apply_rubric(
        product.get("nutriscore_grade"),
        product.get("nova_group"),
        additive_codes,
    )
    verdict_summary = build_verdict_summary(product, product_record["nutrition"], rubric_verdict, rubric_reasons)
    result = {
        "ingredients": ingredients,
        "verdicts": {ingredient: "unknown" for ingredient in ingredients},
        "matches": {ingredient: None for ingredient in ingredients},
        "unknowns": list(ingredients),
        "product": product,
        "nutriscore_grade": product.get("nutriscore_grade"),
        "nutrition": product_record["nutrition"],
        "additives_highlights": _extract_additives_and_highlights(product),
        "flagged_additives": flagged_additives,
        "rubric_verdict": rubric_verdict,
        "rubric_reasons": rubric_reasons,
        "verdict_summary": verdict_summary,
        "explanation": None,
        "violations": [],
        "external_source": product_record["source"],
    }

    client = get_openai_client()
    if skip_llm or client is None:
        result["explanation"] = verdict_summary
        return result

    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": _health_prompt(product, ingredients, result["nutriscore_grade"])}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        assessment = json.loads(response.choices[0].message.content)
        model_verdicts = assessment.get("verdicts", {})
        result["verdicts"] = {
            ingredient: model_verdicts.get(ingredient, "unknown")
            if model_verdicts.get(ingredient, "unknown") in VALID_VERDICTS
            else "unknown"
            for ingredient in ingredients
        }
        result["unknowns"] = [ingredient for ingredient, verdict in result["verdicts"].items() if verdict == "unknown"]
        notes = assessment.get("ingredient_notes", {})
        summary = assessment.get("summary", "No health assessment was returned.")
        if isinstance(notes, dict) and notes:
            summary += "\n\n" + "\n".join(f"{ingredient}: {notes.get(ingredient, 'No explanation returned.')}" for ingredient in ingredients)
        violations = check_output(summary)
        result["llm_explanation"] = "Assessment withheld because it contained medical-claim language." if violations else summary
        result["explanation"] = verdict_summary
        result["violations"] = violations
    except Exception as exc:
        result["llm_explanation"] = f"(Health assessment unavailable: {exc})"
        result["explanation"] = verdict_summary

    return result


def get_chat_model():
    """Return a LangChain chat model only when the API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=api_key, temperature=0) if api_key else None


def _history_to_messages(history):
    """Convert {"role", "content"} turns from the frontend into LangChain message objects."""
    messages = []
    for turn in (history or [])[-8:]:
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _product_context_block(product, ingredients, nutrition, nutriscore_grade):
    """Curated (not raw) product fields, so untrusted crowd-sourced text can't smuggle prompt instructions."""
    curated = {
        "name": product.get("product_name") or product.get("product_name_en") or "Unknown",
        "brand": product.get("brands", ""),
        "category": product.get("categories_tags", ""),
        "nutriscore_grade": nutriscore_grade or "not available",
        "nova_group": product.get("nova_group", "unknown"),
        "ingredients": ingredients,
        "nutrition_per_100g": nutrition,
    }
    return json.dumps(curated, ensure_ascii=False)


def _recent_scans_block(recent_scans):
    """Render the customer's other recently scanned products, for disambiguation only."""
    items = [item for item in (recent_scans or []) if isinstance(item, dict)][:10]
    if not items:
        return "No other recent scans."
    lines = []
    for item in items:
        name = str(item.get("productName") or "Unknown product")
        brand = str(item.get("brand") or "")
        code = str(item.get("barcode") or "")
        lines.append(f"- {name} ({brand}) — barcode {code}".replace(" ()", ""))
    return "\n".join(lines)


_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Classify the customer's food/nutrition question into exactly one label.\n"
        "PRODUCT: only about the specific product currently scanned — triggers like 'this product', "
        "'does this have…', 'is this vegan', 'how much sugar', comparisons across recent scans, verdict "
        "explanations.\n"
        "GENERAL: only general food-science knowledge that does not depend on the scanned product — "
        "triggers like 'what is <ingredient>', 'what is E<number>', 'what does <additive> do', 'how does "
        "Nutri-Score work', 'what's the difference between NOVA 3 and 4', common allergen questions not "
        "tied to a specific product.\n"
        "MIXED: asks a general food-science question AND whether/how it applies to the scanned product in "
        "the same message, e.g. 'what is E471 and is it in this?'.\n"
        "Respond with exactly one word: PRODUCT, GENERAL, or MIXED."
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

_PRODUCT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are Ingrid, a grounded food-product assistant. The customer is asking about ONE specific "
        "product: the one currently open in the app.\n\n"
        "Answer ONLY using the data inside <product_context>. <recent_scans> is provided solely so you "
        "can recognise if the customer is asking about a *different* product they scanned earlier — if so, "
        "tell them you can only discuss the product currently open, and that they should open that other "
        "product's page to ask about it there.\n\n"
        "Rules:\n"
        "- Do not use outside knowledge to fill in facts about this product; if <product_context> does not "
        "cover the question, say so plainly instead of guessing.\n"
        "- Do not state facts about any product other than the one in <product_context>.\n"
        "- Do not give medical, dietary, or treatment advice, and never say a product is safe or unsafe for "
        "a person or condition.\n"
        "- Keep answers to two to four plain-language sentences.\n\n"
        "<product_context>\n{product_context}\n</product_context>\n\n"
        "<recent_scans>\n{recent_scans}\n</recent_scans>"
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

_GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are Ingrid, a food-science assistant. Answer the customer's general food-science question: "
        "what an ingredient or additive IS or DOES, how Nutri-Score or NOVA work, common allergens, or "
        "cooking chemistry.\n\n"
        "Rules:\n"
        "- Answer from general food-science knowledge, not from any specific scanned product.\n"
        "- Do NOT invent or imply facts about the customer's scanned product in this mode.\n"
        "- Answer in two to four sentences. Factual and neutral. Prefer 'generally' / 'typically' / "
        "'commonly used as' phrasing.\n"
        "- Do NOT claim safety verdicts stronger than 'considered safe by regulators' or 'some studies "
        "have raised questions about X'.\n"
        "- Do not give medical, dietary, or treatment advice, and never say something is safe or unsafe for "
        "a person or condition."
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

_MIXED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "The customer's question mixes a GENERAL food-science part with a PRODUCT-specific part about "
        "the product currently open in the app. Answer in two clearly separated parts, in this order, and "
        "make the switch between them visible, for example: 'E471 is <general answer>. Looking at this "
        "product's ingredient list: <grounded answer>.'\n\n"
        "Part 1 — General (from your own food-science knowledge):\n"
        "- Factual and neutral. Prefer 'generally' / 'typically' / 'commonly used as' phrasing.\n"
        "- Do NOT claim safety verdicts stronger than 'considered safe by regulators' or 'some studies "
        "have raised questions about X'.\n\n"
        "Part 2 — Product-grounded (ONLY using <product_context>):\n"
        "- <recent_scans> is provided solely so you can recognise a *different* scanned product — if the "
        "question is actually about one of those, say you can only discuss the product currently open.\n"
        "- If <product_context> does not cover the product part, say so plainly instead of guessing.\n\n"
        "Never give medical, dietary, or treatment advice, and never say a product is safe or unsafe for a "
        "person or condition.\n\n"
        "<product_context>\n{product_context}\n</product_context>\n\n"
        "<recent_scans>\n{recent_scans}\n</recent_scans>"
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def _classify_intent(model, question, history_messages):
    try:
        label = (_CLASSIFY_PROMPT | model | StrOutputParser()).invoke(
            {"question": question, "history": history_messages}
        ).strip().upper()
    except Exception:
        return "PRODUCT"
    if label.startswith("MIXED"):
        return "MIXED"
    if label.startswith("GENERAL"):
        return "GENERAL"
    return "PRODUCT"


def _fetch_product_context(barcode, recent_scans):
    """Retrieve and curate grounding context for the currently scanned barcode, if any."""
    product_record = lookup_pinecone_barcode(barcode) or lookup_openfoodfacts_barcode(barcode)
    if not product_record:
        return "No retrieved data is available for this barcode.", _recent_scans_block(recent_scans)
    product = product_record["details"]
    ingredients = parse_ingredients(product_record["ingredients_text"])
    context = _product_context_block(product, ingredients, product_record["nutrition"], product.get("nutriscore_grade"))
    return context, _recent_scans_block(recent_scans)


def answer_question(barcode, question, history=None, recent_scans=None):
    """Route a chat question to a product-grounded or general-knowledge LangChain answerer."""
    question = str(question or "").strip()
    if not question:
        return {"answer": "Ask a question to get started.", "violations": [], "mode": None}

    model = get_chat_model()
    if model is None:
        return {"answer": "Chat is unavailable because no OpenAI API key is configured.", "violations": [], "mode": None}

    history_messages = _history_to_messages(history)
    intent = _classify_intent(model, question, history_messages)

    if intent == "GENERAL":
        try:
            answer = (_GENERAL_PROMPT | model | StrOutputParser()).invoke(
                {"history": history_messages, "question": question}
            ).strip()
        except Exception as exc:
            return {"answer": f"(Chat unavailable: {exc})", "violations": [], "mode": "general"}
        violations = check_output(answer)
        if violations:
            answer = "I can't answer that directly since it touches on medical or dietary advice. Please consult a healthcare professional."
        return {"answer": answer, "violations": violations, "mode": "general"}

    if intent == "MIXED":
        product_context, recent_scans_text = _fetch_product_context(barcode, recent_scans)
        try:
            answer = (_MIXED_PROMPT | model | StrOutputParser()).invoke({
                "product_context": product_context,
                "recent_scans": recent_scans_text,
                "history": history_messages,
                "question": question,
            }).strip()
        except Exception as exc:
            return {"answer": f"(Chat unavailable: {exc})", "violations": [], "mode": "mixed"}
        violations = check_output(answer)
        if violations:
            answer = "I can't answer that directly since it touches on medical or dietary advice. Please consult a healthcare professional."
        return {"answer": answer, "violations": violations, "mode": "mixed"}

    # PRODUCT mode: retrieve grounding context for the currently scanned barcode (Pinecone, falling back to OpenFoodFacts)
    product_record = lookup_pinecone_barcode(barcode) or lookup_openfoodfacts_barcode(barcode)
    if not product_record:
        return {
            "answer": f"I don't have any retrieved data for barcode {barcode or 'unknown'}. Scan the product first.",
            "violations": [],
            "mode": "product",
        }

    product = product_record["details"]
    ingredients = parse_ingredients(product_record["ingredients_text"])
    product_context = _product_context_block(product, ingredients, product_record["nutrition"], product.get("nutriscore_grade"))

    try:
        answer = (_PRODUCT_PROMPT | model | StrOutputParser()).invoke({
            "product_context": product_context,
            "recent_scans": _recent_scans_block(recent_scans),
            "history": history_messages,
            "question": question,
        }).strip()
    except Exception as exc:
        return {"answer": f"(Chat unavailable: {exc})", "violations": [], "mode": "product"}

    violations = check_output(answer)
    if violations:
        answer = "I can't answer that directly since it touches on medical or dietary advice. Please consult a healthcare professional."
    return {"answer": answer, "violations": violations, "mode": "product"}
