import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import chain


# ---------------------------------------------------------------------------
# parse_ingredients
# ---------------------------------------------------------------------------

def test_parse_ingredients_basic_comma_list():
    assert chain.parse_ingredients("Water, Sugar, Salt") == ["water", "sugar", "salt"]


def test_parse_ingredients_strips_leading_label_and_dedupes():
    assert chain.parse_ingredients("Ingredients: Water, Water, Sugar") == ["water", "sugar"]


def test_parse_ingredients_handles_parens_semicolons_and_newlines():
    text = "Milk (Pasteurized), Sugar; Cocoa\nSoy Lecithin"
    assert chain.parse_ingredients(text) == ["milk", "pasteurized", "sugar", "cocoa", "soy lecithin"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_parse_ingredients_empty_input_returns_empty_list(value):
    assert chain.parse_ingredients(value) == []


# ---------------------------------------------------------------------------
# _dedupe_ingredients_text
# ---------------------------------------------------------------------------

def test_dedupe_ingredients_text_collapses_repeats():
    assert chain._dedupe_ingredients_text("Water; Water; Sugar") == "Water; Sugar"


@pytest.mark.parametrize("value", [None, ""])
def test_dedupe_ingredients_text_empty_input(value):
    assert chain._dedupe_ingredients_text(value) == ""


# ---------------------------------------------------------------------------
# _extract_nutrition
# ---------------------------------------------------------------------------

def test_extract_nutrition_parses_embedded_json():
    text = (
        'Nutrition per 100g: [{"name": "sugars", "100g": 10.5, "unit": "g"}, '
        '{"name": "salt", "100g": 0.1}]\nNutri-Score: e'
    )
    assert chain._extract_nutrition(text) == {
        "sugars": {"value": 10.5, "unit": "g"},
        "salt": {"value": 0.1, "unit": ""},
    }


def test_extract_nutrition_no_match_returns_empty_dict():
    assert chain._extract_nutrition("no nutrition info here") == {}


def test_extract_nutrition_malformed_json_returns_empty_dict():
    text = "Nutrition per 100g: [not valid json}\nNutri-Score: e"
    assert chain._extract_nutrition(text) == {}


def test_extract_nutrition_skips_rows_missing_name_or_value():
    text = 'Nutrition per 100g: [{"name": "", "100g": 5}, {"name": "fat"}]\nNutri-Score: a'
    assert chain._extract_nutrition(text) == {}


# ---------------------------------------------------------------------------
# _extract_additives_and_highlights
# ---------------------------------------------------------------------------

def test_extract_additives_and_highlights_splits_e_numbers_from_tags():
    metadata = {"ingredients_tags": "en:e150a;en:e338;en:water;en:high-fructose-corn-syrup"}
    result = chain._extract_additives_and_highlights(metadata)
    assert result["additives"] == ["E150A", "E338"]
    assert result["highlights"] == ["Water", "High Fructose Corn Syrup"]


def test_extract_additives_and_highlights_missing_tags_key():
    assert chain._extract_additives_and_highlights({}) == {"additives": [], "highlights": []}


def test_extract_additives_and_highlights_caps_highlights_at_twelve():
    tags = ";".join(f"en:tag-{i}" for i in range(20))
    result = chain._extract_additives_and_highlights({"ingredients_tags": tags})
    assert len(result["highlights"]) == 12


# ---------------------------------------------------------------------------
# check_output (banned medical-claim phrase guardrail)
# ---------------------------------------------------------------------------

def test_check_output_flags_banned_phrase():
    assert chain.check_output("This treats your headache") == ["treats your"]


def test_check_output_is_case_insensitive():
    assert chain.check_output("IT CURES EVERYTHING") == ["cures"]


def test_check_output_clean_text_returns_no_violations():
    assert chain.check_output("This product is a fizzy drink.") == []


def test_check_output_can_flag_multiple_phrases():
    violations = chain.check_output("This cures and diagnoses your condition")
    assert "cures" in violations
    assert "diagnos" in violations


# ---------------------------------------------------------------------------
# _product_context_block / _recent_scans_block
# ---------------------------------------------------------------------------

def test_product_context_block_curates_expected_fields():
    product = {"product_name": "Coke", "brands": "Coca-Cola", "categories_tags": "beverages", "nova_group": 4}
    block = chain._product_context_block(product, ["water", "sugar"], {"sugars": {"value": 10, "unit": "g"}}, "e")
    assert json.loads(block) == {
        "name": "Coke",
        "brand": "Coca-Cola",
        "category": "beverages",
        "nutriscore_grade": "e",
        "nova_group": 4,
        "ingredients": ["water", "sugar"],
        "nutrition_per_100g": {"sugars": {"value": 10, "unit": "g"}},
    }


def test_product_context_block_falls_back_when_fields_missing():
    block = json.loads(chain._product_context_block({}, [], {}, None))
    assert block["name"] == "Unknown"
    assert block["nutriscore_grade"] == "not available"
    assert block["nova_group"] == "unknown"


def test_recent_scans_block_empty():
    assert chain._recent_scans_block([]) == "No other recent scans."
    assert chain._recent_scans_block(None) == "No other recent scans."


def test_recent_scans_block_formats_entries_and_ignores_non_dicts():
    scans = [
        {"productName": "Nutella", "brand": "Ferrero", "barcode": "3017620422003"},
        "not-a-dict",
        {"productName": "Water", "barcode": "123"},
    ]
    block = chain._recent_scans_block(scans)
    lines = block.split("\n")
    assert lines[0] == "- Nutella (Ferrero) — barcode 3017620422003"
    assert lines[1] == "- Water — barcode 123"


def test_recent_scans_block_caps_at_ten_entries():
    scans = [{"productName": f"Product {i}", "barcode": str(i)} for i in range(15)]
    block = chain._recent_scans_block(scans)
    assert len(block.split("\n")) == 10


# ---------------------------------------------------------------------------
# read_barcode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   "])
def test_read_barcode_empty_input(value):
    assert chain.read_barcode(value) == ""


def test_read_barcode_passes_through_raw_code():
    assert chain.read_barcode("04904403") == "04904403"


def test_read_barcode_coerces_non_string_input():
    assert chain.read_barcode(4904403) == "4904403"


# ---------------------------------------------------------------------------
# lookup_pinecone_barcode / lookup_openfoodfacts_barcode — input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("barcode", ["abc", "123", "not-a-barcode", ""])
def test_lookup_pinecone_barcode_rejects_invalid_format(barcode, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "fake-key")
    assert chain.lookup_pinecone_barcode(barcode) is None


def test_lookup_pinecone_barcode_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    assert chain.lookup_pinecone_barcode("04904403") is None


@pytest.mark.parametrize("barcode", ["abc", "123", ""])
def test_lookup_openfoodfacts_barcode_rejects_invalid_format(barcode):
    assert chain.lookup_openfoodfacts_barcode(barcode) is None


class _FakeVector:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeFetchResult:
    def __init__(self, vectors):
        self.vectors = vectors


class _FakeIndex:
    def __init__(self, vectors_by_id):
        self._vectors_by_id = vectors_by_id

    def fetch(self, ids):
        return _FakeFetchResult({i: self._vectors_by_id[i] for i in ids if i in self._vectors_by_id})


class _FakePinecone:
    def __init__(self, api_key):
        self.api_key = api_key

    def Index(self, name):
        return _FakeIndex({
            "12345678": _FakeVector({
                "product_name": "Test Product",
                "ingredients_text": "Water; Water; Sugar",
                "text": 'Nutrition per 100g: [{"name": "sugars", "100g": 5, "unit": "g"}]\nNutri-Score: c',
            })
        })


def test_lookup_pinecone_barcode_returns_curated_record(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "fake-key")
    monkeypatch.setattr(chain, "Pinecone", _FakePinecone)
    record = chain.lookup_pinecone_barcode("12345678")
    assert record["ingredients_text"] == "Water; Sugar"
    assert record["nutrition"] == {"sugars": {"value": 5, "unit": "g"}}
    assert record["source"] == f"Pinecone: {chain.PRODUCT_INDEX_NAME}"


def test_lookup_pinecone_barcode_missing_vector_returns_none(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "fake-key")
    monkeypatch.setattr(chain, "Pinecone", _FakePinecone)
    assert chain.lookup_pinecone_barcode("99999999") is None


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_lookup_openfoodfacts_barcode_not_found(monkeypatch):
    monkeypatch.setattr(chain.requests, "get", lambda *a, **k: _FakeResponse({"status": 0}))
    assert chain.lookup_openfoodfacts_barcode("04904403") is None


def test_lookup_openfoodfacts_barcode_found(monkeypatch):
    payload = {
        "status": 1,
        "product": {
            "product_name": "Coke",
            "ingredients_text": "Water, Sugar",
            "categories_tags": ["en:beverages"],
            "ingredients_tags": ["en:water"],
            "additives_tags": ["en:e150a"],
            "nutriments": {"sugars_100g": 10.5, "sugars_unit": "g"},
        },
    }
    monkeypatch.setattr(chain.requests, "get", lambda *a, **k: _FakeResponse(payload))
    record = chain.lookup_openfoodfacts_barcode("04904403")
    assert record["ingredients_text"] == "Water, Sugar"
    assert record["nutrition"] == {"sugars": {"value": 10.5, "unit": "g"}}
    assert record["source"] == "OpenFoodFacts API"
    assert record["details"]["categories_tags"] == "en:beverages"


# ---------------------------------------------------------------------------
# get_openai_client / get_chat_model
# ---------------------------------------------------------------------------

def test_get_openai_client_none_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert chain.get_openai_client() is None


def test_get_openai_client_returns_client_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    assert chain.get_openai_client() is not None


def test_get_chat_model_none_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert chain.get_chat_model() is None


def test_get_chat_model_returns_model_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    assert chain.get_chat_model() is not None


# ---------------------------------------------------------------------------
# _history_to_messages
# ---------------------------------------------------------------------------

def test_history_to_messages_converts_and_ignores_unknown_roles():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "  "},
    ]
    messages = chain._history_to_messages(history)
    assert messages == [HumanMessage(content="hi"), AIMessage(content="hello")]


def test_history_to_messages_keeps_only_last_eight():
    history = [{"role": "user", "content": str(i)} for i in range(12)]
    messages = chain._history_to_messages(history)
    assert len(messages) == 8
    assert [m.content for m in messages] == [str(i) for i in range(4, 12)]


# ---------------------------------------------------------------------------
# _classify_intent
# ---------------------------------------------------------------------------

def _model_returning(content):
    def fake_model(_prompt_value):
        return AIMessage(content=content)
    return fake_model


def _raising_model(_prompt_value):
    raise RuntimeError("boom")


@pytest.mark.parametrize("label, expected", [
    ("GENERAL", "GENERAL"),
    ("mixed", "MIXED"),
    ("PRODUCT", "PRODUCT"),
    ("something unexpected", "PRODUCT"),
])
def test_classify_intent_parses_label(label, expected):
    assert chain._classify_intent(_model_returning(label), "question", []) == expected


def test_classify_intent_defaults_to_product_on_error():
    assert chain._classify_intent(_raising_model, "question", []) == "PRODUCT"


# ---------------------------------------------------------------------------
# answer_question — end-to-end routing (LLM and lookups mocked)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["", "   "])
def test_answer_question_empty_question(question):
    result = chain.answer_question("04904403", question)
    assert result == {"answer": "Ask a question to get started.", "violations": [], "mode": None}


def test_answer_question_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = chain.answer_question("04904403", "How much sugar?")
    assert result["mode"] is None
    assert "no OpenAI API key is configured" in result["answer"]


def test_answer_question_general_mode_skips_product_lookup(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(chain, "get_chat_model", lambda: _model_returning("General mocked answer"))
    monkeypatch.setattr(chain, "_classify_intent", lambda *a, **k: "GENERAL")

    def _fail_if_called(*_a, **_k):
        raise AssertionError("product lookup should not run in GENERAL mode")

    monkeypatch.setattr(chain, "lookup_pinecone_barcode", _fail_if_called)
    monkeypatch.setattr(chain, "lookup_openfoodfacts_barcode", _fail_if_called)

    result = chain.answer_question("04904403", "What does NOVA group 4 mean?")
    assert result == {"answer": "General mocked answer", "violations": [], "mode": "general"}


def test_answer_question_product_mode_found(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(chain, "get_chat_model", lambda: _model_returning("Product mocked answer"))
    monkeypatch.setattr(chain, "_classify_intent", lambda *a, **k: "PRODUCT")
    monkeypatch.setattr(chain, "lookup_pinecone_barcode", lambda barcode: {
        "details": {"product_name": "Coke"},
        "ingredients_text": "water, sugar",
        "nutrition": {},
        "source": "Pinecone: ingrid-beverages",
    })

    result = chain.answer_question("04904403", "How much sugar is in this?")
    assert result == {"answer": "Product mocked answer", "violations": [], "mode": "product"}


def test_answer_question_product_mode_not_found_skips_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(chain, "_classify_intent", lambda *a, **k: "PRODUCT")
    monkeypatch.setattr(chain, "lookup_pinecone_barcode", lambda barcode: None)
    monkeypatch.setattr(chain, "lookup_openfoodfacts_barcode", lambda barcode: None)

    def _fail_if_invoked(_prompt_value):
        raise AssertionError("model should not be invoked when there is no product data")

    monkeypatch.setattr(chain, "get_chat_model", lambda: _fail_if_invoked)

    result = chain.answer_question("00000000", "How much sugar is in this?")
    assert result["mode"] == "product"
    assert "I don't have any retrieved data" in result["answer"]


def test_answer_question_mixed_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(chain, "get_chat_model", lambda: _model_returning("Mixed mocked answer"))
    monkeypatch.setattr(chain, "_classify_intent", lambda *a, **k: "MIXED")
    monkeypatch.setattr(chain, "lookup_pinecone_barcode", lambda barcode: {
        "details": {"product_name": "Coke"},
        "ingredients_text": "water, sugar",
        "nutrition": {},
        "source": "Pinecone: ingrid-beverages",
    })

    result = chain.answer_question("04904403", "What is E150a and is it in this?")
    assert result == {"answer": "Mixed mocked answer", "violations": [], "mode": "mixed"}


def test_answer_question_applies_guardrail_on_banned_phrase(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(chain, "get_chat_model", lambda: _model_returning("This treats your indigestion"))
    monkeypatch.setattr(chain, "_classify_intent", lambda *a, **k: "GENERAL")

    result = chain.answer_question("04904403", "Does this cure indigestion?")
    assert result["violations"] == ["treats your"]
    assert "consult a healthcare professional" in result["answer"]


def test_answer_question_model_error_is_reported(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(chain, "get_chat_model", lambda: _raising_model)
    monkeypatch.setattr(chain, "_classify_intent", lambda *a, **k: "GENERAL")

    result = chain.answer_question("04904403", "What is E471?")
    assert result["answer"].startswith("(Chat unavailable:")
    assert result["mode"] == "general"
