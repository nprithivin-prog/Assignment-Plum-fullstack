"""
Run with:  python3 -m pytest test_profiler.py -q
"""

from __future__ import annotations

import pytest

from profiler import (
    GuardrailExit,
    build_recommendations,
    classify_risk,
    extract_factors,
    ingest,
    make_sample_form,
    parse_survey,
    run_pipeline,
)

REFERENCE = {"age": 42, "smoker": True, "exercise": "rarely", "diet": "high sugar"}
OCR_TEXT = "Age: 42\nSmoker: yes\nExercise: rarely\nDiet: high sugar"


# --------------------------------------------------------------------------
# Step 1 - parsing
# --------------------------------------------------------------------------

def test_typed_json_matches_contract():
    result = parse_survey(REFERENCE, source="json").to_dict()
    assert result["answers"] == REFERENCE
    assert result["missing_fields"] == []
    assert result["confidence"] > 0.9


def test_ocr_text_matches_typed_json():
    assert ingest(OCR_TEXT).answers == REFERENCE


def test_json_string_payload_is_detected():
    assert ingest('{"age":42,"smoker":true,"exercise":"rarely","diet":"high sugar"}').answers \
        == REFERENCE


@pytest.mark.parametrize("text,expected_age", [
    ("Age: 42", 42),
    ("age 42", 42),
    ("Patient Age = 42", 42),
    ("Age in years : 42", 42),
])
def test_age_label_variants(text, expected_age):
    assert ingest(text).answers.get("age") == expected_age


def test_ocr_typos_in_labels_are_recovered():
    noisy = "Aqe: 42\nSrnoker: yes\nExercse: rarely\nDlet: high sugar"
    answers = ingest(noisy).answers
    assert answers.get("smoker") is True
    assert answers.get("exercise") == "rarely"
    assert answers.get("diet") == "high sugar"


def test_negation_is_not_swallowed_by_substring_match():
    # "non-smoker" contains "smoker", a positive token. Order of checks matters.
    assert ingest("Smoker: non-smoker").answers["smoker"] is False
    assert ingest("Smoker: never").answers["smoker"] is False
    assert ingest({"smoker": "No"}).answers["smoker"] is False


def test_nullish_values_count_as_missing():
    result = ingest("Age: 42\nSmoker: N/A\nExercise: -\nDiet: high sugar")
    assert "smoker" in result.missing_fields
    assert "exercise" in result.missing_fields
    assert result.answers["age"] == 42


def test_multiple_fields_on_one_line():
    answers = ingest("Age: 42   Smoker: yes\nExercise: rarely   Diet: high sugar").answers
    assert answers == REFERENCE


def test_unparseable_value_is_missing_not_guessed():
    result = ingest({"age": "abc", "smoker": True, "exercise": "rarely", "diet": "high sugar"})
    assert "age" in result.missing_fields
    assert "age" not in result.answers


def test_confidence_drops_when_fields_are_missing():
    full = ingest(OCR_TEXT).confidence
    partial = ingest("Age: 42\nSmoker: yes\nExercise: rarely").confidence
    assert partial < full


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------

def test_guardrail_fires_above_half_missing():
    with pytest.raises(GuardrailExit) as exc:
        run_pipeline({"age": 42})
    payload = exc.value.payload
    assert payload["status"] == "incomplete_profile"
    assert payload["reason"] == ">50% fields missing"
    assert set(payload["missing_fields"]) == {"smoker", "exercise", "diet"}


def test_exactly_half_missing_is_allowed():
    # 2 of 4 missing is not ">50%", so the profile must still be produced.
    result = run_pipeline({"age": 42, "smoker": True})
    assert result["status"] == "ok"


def test_empty_input_is_rejected():
    with pytest.raises(GuardrailExit):
        run_pipeline({})


def test_garbage_input_is_rejected():
    with pytest.raises(GuardrailExit):
        run_pipeline("!!! ??? ***")


# --------------------------------------------------------------------------
# Step 2 - factors
# --------------------------------------------------------------------------

def test_reference_factors():
    parsed = ingest(REFERENCE)
    result = extract_factors(parsed.answers, parsed.confidence,
                             parsed.field_confidence, parsed.qualifiers)
    assert result.factors == ["smoking", "poor diet", "low exercise"]
    assert result.age_factor == "age 40-49"
    assert 0.8 <= result.confidence <= 0.99


def test_healthy_profile_has_no_factors():
    parsed = ingest({"age": 25, "smoker": False, "exercise": "daily", "diet": "balanced"})
    assert extract_factors(parsed.answers, parsed.confidence,
                           parsed.field_confidence, parsed.qualifiers).factors == []


def test_occasional_smoking_is_graded_lower():
    parsed = ingest("Smoker: occasionally\nAge: 30\nExercise: daily\nDiet: balanced")
    factors = extract_factors(parsed.answers, parsed.confidence,
                              parsed.field_confidence, parsed.qualifiers)
    assert "occasional smoking" in factors.factors
    assert "smoking" not in factors.factors


def test_never_exercising_is_worse_than_rarely():
    low = classify_risk(["low exercise"]).score
    none = classify_risk(["sedentary lifestyle"]).score
    assert none > low


def test_optional_fields_add_factors():
    parsed = ingest({"age": 45, "smoker": False, "exercise": "daily",
                     "diet": "balanced", "bmi": 32, "sleep_hours": 5})
    factors = extract_factors(parsed.answers, parsed.confidence,
                              parsed.field_confidence, parsed.qualifiers).factors
    assert "obesity" in factors and "insufficient sleep" in factors


# --------------------------------------------------------------------------
# Step 3 - risk
# --------------------------------------------------------------------------

def test_reference_score_and_level():
    risk = classify_risk(["smoking", "poor diet", "low exercise"], age_factor="age 40-49")
    assert risk.score == 78
    assert risk.risk_level == "high"
    assert risk.rationale == ["smoking", "high sugar diet", "low activity"]


def test_score_is_capped_at_100():
    risk = classify_risk(list({"smoking", "poor diet", "sedentary lifestyle",
                               "heavy alcohol use", "obesity", "family history"}),
                         age_factor="age 70+")
    assert risk.score == 100


def test_clean_profile_is_low_risk():
    result = run_pipeline({"age": 24, "smoker": False, "exercise": "daily", "diet": "balanced"})
    assert result["risk_level"] == "low"
    assert result["factors"] == []


def test_levels_are_monotonic_in_score():
    order = {"low": 0, "moderate": 1, "high": 2}
    previous = 0
    for factors in ([], ["low exercise"], ["poor diet", "low exercise"],
                    ["smoking", "poor diet", "low exercise"]):
        level = order[classify_risk(factors).risk_level]
        assert level >= previous
        previous = level


# --------------------------------------------------------------------------
# Step 4 - recommendations
# --------------------------------------------------------------------------

def test_reference_recommendations_lead_with_headlines():
    recs = build_recommendations(["smoking", "poor diet", "low exercise"], "high")
    assert recs[:3] == ["Quit smoking", "Reduce sugar", "Walk 30 mins daily"]


def test_recommendations_are_ordered_by_impact():
    recs = build_recommendations(["low exercise", "smoking"], "high")
    assert recs[0] == "Quit smoking"


def test_no_factors_yields_maintenance_advice():
    recs = build_recommendations([], "low")
    assert recs and all(isinstance(r, str) for r in recs)


def test_recommendations_are_deduplicated_and_capped():
    recs = build_recommendations(
        ["smoking", "poor diet", "low exercise", "obesity", "family history",
         "heavy alcohol use", "insufficient sleep"], "high")
    assert len(recs) == len(set(recs)) <= 5


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_full_contract_shape():
    result = run_pipeline(REFERENCE)
    assert result["risk_level"] == "high"
    assert result["factors"] == ["smoking", "poor diet", "low exercise"]
    assert result["recommendations"][:3] == ["Quit smoking", "Reduce sugar",
                                             "Walk 30 mins daily"]
    assert result["status"] == "ok"


def test_text_and_json_inputs_agree():
    assert run_pipeline(OCR_TEXT)["risk_level"] == run_pipeline(REFERENCE)["risk_level"]


def test_scanned_image_round_trip(tmp_path):
    pytest.importorskip("pytesseract")

    path = make_sample_form(str(tmp_path / "form.png"))
    with open(path, "rb") as handle:
        result = run_pipeline(image_bytes=handle.read())
    assert result["risk_level"] == "high"
    assert "smoking" in result["factors"]
