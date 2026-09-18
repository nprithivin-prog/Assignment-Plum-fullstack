#!/usr/bin/env python3
"""
AI-Powered Health Risk Profiler
===============================

A single-file service that turns a lifestyle survey - typed, pasted as text, or
photographed - into a structured risk profile: factors, a score, and actionable
guidance.

Pipeline:  OCR/parse  ->  factor extraction  ->  risk scoring  ->  recommendations

Deliberately rule-based rather than generative. A health screening score has to
be explainable: every point traces back to a named factor, and every
recommendation traces back to a factor actually present in the answers.
NOTHING HERE IS DIAGNOSTIC.

Run the API + demo page:
    pip install fastapi uvicorn python-multipart pillow pytesseract
    sudo apt-get install tesseract-ocr          # only needed for the image path
    uvicorn profiler:app --reload               # http://localhost:8000/demo

Run it from the command line instead:
    python profiler.py --json '{"age":42,"smoker":true,"exercise":"rarely","diet":"high sugar"}'
    python profiler.py --image form.png --verbose
    cat survey.txt | python profiler.py

Generate a scan-like test form:
    python profiler.py --make-sample form.png

FastAPI is optional: the parsing, scoring and CLI all work without it.

Sections below, in order:
    1. Domain configuration      - fields, vocabularies, weights, advice
    2. OCR                       - image -> text
    3. Parsing                   - text/JSON -> answers
    4. Analysis                  - factors, score, recommendations
    5. Orchestration             - pipeline + guardrails
    6. HTTP API                  - FastAPI routes
    7. Demo page                 - embedded browser client
    8. Sample form generator     - renders a scan-like image for testing
    9. CLI
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps

    OCR_AVAILABLE = True
except ImportError:                       # pragma: no cover - no OCR engine present
    OCR_AVAILABLE = False

try:
    from fastapi import Body, FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:                       # pragma: no cover - CLI-only install
    FASTAPI_AVAILABLE = False


# ==========================================================================
# DOMAIN CONFIGURATION
# Everything a clinician or product owner would tune lives in this section:
# which fields we ask for, how noisy values map to canonical ones, what each
# risk factor is worth, and what advice it earns. Nothing here is diagnostic.
# ==========================================================================

# --------------------------------------------------------------------------
# Survey schema
# --------------------------------------------------------------------------

# Required fields drive the >50% missing guardrail.
REQUIRED_FIELDS: tuple[str, ...] = ("age", "smoker", "exercise", "diet")

# Optional fields enrich the profile but never trigger the guardrail.
OPTIONAL_FIELDS: tuple[str, ...] = ("alcohol", "sleep_hours", "bmi", "family_history")

ALL_FIELDS: tuple[str, ...] = REQUIRED_FIELDS + OPTIONAL_FIELDS

# Aliases seen on real forms / produced by OCR. Keys are matched case- and
# punctuation-insensitively, then fuzzily (see parser.py) to survive typos.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "age": ("age", "age in years", "years old", "dob age", "patient age"),
    "smoker": ("smoker", "smoking", "do you smoke", "tobacco", "smoking status"),
    "exercise": ("exercise", "physical activity", "activity", "workout",
                 "exercise frequency", "how often do you exercise"),
    "diet": ("diet", "eating habits", "nutrition", "food habits", "diet type"),
    "alcohol": ("alcohol", "drinking", "alcohol consumption", "drinks per week"),
    "sleep_hours": ("sleep", "sleep hours", "hours of sleep", "sleep per night"),
    "bmi": ("bmi", "body mass index"),
    "family_history": ("family history", "hereditary conditions", "family illness"),
}

# --------------------------------------------------------------------------
# Value vocabularies (noisy value -> canonical value)
# --------------------------------------------------------------------------

TRUE_WORDS = ("yes", "y", "true", "1", "yeah", "yep", "daily", "regularly",
              "occasionally", "sometimes", "smoker", "current", "ex-smoker", "former")
# NB: "na"/"n/a" are deliberately absent - they mean "not answered", which is a
# missing field, not a negative answer. They are handled by the NULLISH set.
FALSE_WORDS = ("no", "n", "false", "0", "never", "none", "non smoker",
               "non-smoker", "nonsmoker", "nil")

# Anything in TRUE_WORDS that still deserves a softer score than a daily smoker.
SMOKER_LIGHT_WORDS = ("occasionally", "sometimes", "ex-smoker", "former", "rarely")

# exercise -> ordinal level 0 (never) .. 4 (daily)
EXERCISE_LEVELS: dict[str, int] = {
    "never": 0, "none": 0, "no": 0, "sedentary": 0, "not at all": 0,
    "rarely": 1, "seldom": 1, "once a month": 1, "hardly": 1, "occasionally": 1,
    "sometimes": 2, "weekly": 2, "1-2 times a week": 2, "moderate": 2, "twice a week": 2,
    "often": 3, "regularly": 3, "3-4 times a week": 3, "frequent": 3, "active": 3,
    "daily": 4, "every day": 4, "5+ times a week": 4, "very active": 4,
}

# Diet keywords -> quality bucket. "poor" beats "good" when both appear.
DIET_POOR_KEYWORDS = ("high sugar", "sugary", "sugar", "junk", "fast food", "fried",
                      "processed", "high fat", "high salt", "soda", "oily",
                      "irregular", "skipping meals", "unhealthy", "high carb")
DIET_GOOD_KEYWORDS = ("balanced", "healthy", "home cooked", "homecooked", "vegetarian",
                      "vegan", "mediterranean", "low sugar", "low fat", "high protein",
                      "whole grain", "plenty of vegetables", "fruits and vegetables")

ALCOHOL_HEAVY_KEYWORDS = ("heavy", "daily", "frequent", "excessive", "binge", "everyday")
ALCOHOL_LIGHT_KEYWORDS = ("never", "none", "rarely", "occasionally", "socially", "no")

FAMILY_HISTORY_POSITIVE = ("diabetes", "heart", "cardiac", "cancer", "stroke",
                           "hypertension", "bp", "cholesterol", "yes")

# --------------------------------------------------------------------------
# Scoring model
# --------------------------------------------------------------------------
# Points are additive and capped at 100. Tuned so the reference profile
# (age 42, smoker, rarely exercises, high-sugar diet) lands on 78 = "high".

FACTOR_WEIGHTS: dict[str, int] = {
    "smoking": 30,
    "occasional smoking": 15,
    "poor diet": 20,
    "low exercise": 20,
    "sedentary lifestyle": 25,
    "heavy alcohol use": 15,
    "insufficient sleep": 8,
    "obesity": 15,
    "overweight": 8,
    "family history": 10,
    "age 30-39": 4,
    "age 40-49": 8,
    "age 50-59": 12,
    "age 60-69": 16,
    "age 70+": 20,
}

# Human-readable rationale strings (what the report shows instead of the slug).
FACTOR_RATIONALE: dict[str, str] = {
    "smoking": "smoking",
    "occasional smoking": "occasional smoking",
    "poor diet": "high sugar diet",
    "low exercise": "low activity",
    "sedentary lifestyle": "no regular physical activity",
    "heavy alcohol use": "frequent alcohol use",
    "insufficient sleep": "insufficient sleep",
    "obesity": "obesity (BMI 30+)",
    "overweight": "elevated BMI",
    "family history": "family history of chronic illness",
}

# score -> risk level
RISK_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 25, "low"),
    (26, 55, "moderate"),
    (56, 100, "high"),
)

# --------------------------------------------------------------------------
# Recommendations (non-diagnostic, action-oriented, one line each)
# --------------------------------------------------------------------------

RECOMMENDATIONS: dict[str, list[str]] = {
    "smoking": ["Quit smoking", "Ask a clinician about nicotine replacement or a cessation programme"],
    "occasional smoking": ["Stop occasional smoking before it becomes a daily habit"],
    "poor diet": ["Reduce sugar", "Swap sugary drinks for water or unsweetened tea",
                  "Add a vegetable or fruit portion to two meals a day"],
    "low exercise": ["Walk 30 mins daily", "Build up to 150 minutes of moderate activity per week"],
    "sedentary lifestyle": ["Start with a 10-minute daily walk and increase weekly",
                            "Stand up and move for 5 minutes every hour"],
    "heavy alcohol use": ["Cut alcohol to within national low-risk guidelines",
                          "Keep at least three alcohol-free days each week"],
    "insufficient sleep": ["Aim for 7-8 hours of sleep with a consistent bedtime"],
    "obesity": ["Target a gradual 5-10% weight reduction with a dietitian's help"],
    "overweight": ["Keep weight stable with portion control and regular activity"],
    "family history": ["Share your family history with a clinician and ask about earlier screening"],
}

# Advice added purely because of age band, not a behaviour.
AGE_RECOMMENDATIONS: dict[str, str] = {
    "age 40-49": "Book a routine check-up including blood pressure and blood sugar",
    "age 50-59": "Book a routine check-up including blood pressure, sugar and lipids",
    "age 60-69": "Keep up age-appropriate screening with your clinician",
    "age 70+": "Keep up age-appropriate screening with your clinician",
}

BASELINE_RECOMMENDATIONS: list[str] = [
    "Keep up your current healthy habits",
    "Have a routine check-up once a year",
]

MAX_RECOMMENDATIONS = 5

DISCLAIMER = (
    "This is a lifestyle screening heuristic, not a medical diagnosis. "
    "Consult a qualified clinician for medical advice."
)

# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------

MISSING_FIELD_THRESHOLD = 0.5      # >50% of required fields missing -> abort
MIN_CONFIDENCE = 0.40              # parse confidence below this -> abort

# Confidence multipliers per input channel.
SOURCE_CONFIDENCE: dict[str, float] = {
    "json": 1.00,
    "text": 0.97,
    "image": 0.93,
}


# ==========================================================================
# STEP 1a - IMAGE -> TEXT
# Tesseract is wrapped so the rest of the pipeline never has to know whether
# the text came from a scan or a keyboard.
# ==========================================================================


class OCRUnavailable(RuntimeError):
    """Raised when an image is submitted but no OCR engine is installed."""


@dataclass
class OCRResult:
    text: str
    confidence: float          # 0..1, mean of per-word Tesseract confidences
    word_count: int


# Page segmentation modes worth trying. 4 = single column of variable-size
# text (typical form), 6 = uniform block, 11 = sparse text. Which one wins
# depends on the layout, so we try all three and keep the best read.
PSM_CANDIDATES = (4, 6, 11)

# Below this, upscaling genuinely helps; above it, upscaling just amplifies
# scanner noise and makes Tesseract worse.
UPSCALE_BELOW = 600


def _preprocess(image: "Image.Image") -> "Image.Image":
    """Grayscale, de-speckle, normalise contrast, upscale only when too small."""
    image = image.convert("L")
    image = image.filter(ImageFilter.MedianFilter(3))   # kills salt-and-pepper scan noise
    image = ImageOps.autocontrast(image)
    if min(image.size) < UPSCALE_BELOW:
        scale = min(4, max(2, UPSCALE_BELOW // max(1, min(image.size))))
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    return image


def _read(image: "Image.Image", psm: int) -> OCRResult:
    raw = pytesseract.image_to_data(
        image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )

    confidences: list[float] = []
    lines: dict[tuple[int, int, int], list[str]] = {}

    for i, word in enumerate(raw["text"]):
        word = word.strip()
        if not word:
            continue
        conf = float(raw["conf"][i])
        if conf < 0:          # -1 means "no confidence reported"
            continue
        confidences.append(conf / 100.0)
        key = (raw["block_num"][i], raw["par_num"][i], raw["line_num"][i])
        lines.setdefault(key, []).append(word)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(text=text, confidence=round(mean_conf, 4),
                     word_count=len(confidences))


def _quality(result: OCRResult) -> float:
    """
    Rank candidate reads. Confidence alone is gameable - a pass that finds two
    crisp words beats one that finds twenty good ones - so reward coverage too,
    with diminishing returns.
    """
    if result.word_count == 0:
        return 0.0
    coverage = min(result.word_count, 40) / 40
    return result.confidence * (0.7 + 0.3 * coverage)


def image_to_text(data: bytes) -> OCRResult:
    """Run OCR over raw image bytes, keeping the best of several passes."""
    if not OCR_AVAILABLE:
        raise OCRUnavailable(
            "OCR engine not installed. Install tesseract-ocr and pytesseract, "
            "or send the survey as text/JSON instead."
        )

    image = _preprocess(Image.open(io.BytesIO(data)))

    best: OCRResult | None = None
    for psm in PSM_CANDIDATES:
        try:
            candidate = _read(image, psm)
        except pytesseract.TesseractError:
            continue
        if best is None or _quality(candidate) > _quality(best):
            best = candidate

    return best or OCRResult(text="", confidence=0.0, word_count=0)


# ==========================================================================
# STEP 1b - TEXT/JSON -> ANSWERS + MISSING FIELDS + CONFIDENCE
# ==========================================================================


KEY_MATCH_CUTOFF = 0.72           # fuzzy ratio below this is not a field label
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_LINE_SPLIT = re.compile(r"[\n\r]+|(?<=[a-z0-9])\s{3,}")
_KV_SPLIT = re.compile(r"\s*[:=\-–—]\s*|\s{2,}")

NULLISH = {"", "-", "--", "n/a", "na", "none given", "not stated", "unknown",
           "null", "?", "blank", "not provided", "not answered", "no response"}
# Compared after punctuation/space removal so "N/A", "n / a" and "na" all match.
_NULLISH_SQUASHED = {_s.replace("/", "").replace(" ", "") for _s in NULLISH}


@dataclass
class ParseResult:
    answers: dict[str, Any]
    missing_fields: list[str]
    confidence: float
    field_confidence: dict[str, float] = field(default_factory=dict)
    source: str = "json"
    unrecognised: list[str] = field(default_factory=list)
    # Free-text qualifiers kept aside so `answers` stays contract-clean,
    # e.g. {"smoker": "occasionally"} -> graded rather than binary scoring.
    qualifiers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answers": self.answers,
            "missing_fields": self.missing_fields,
            "confidence": self.confidence,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return _PUNCT.sub(" ", str(s).strip().lower()).strip()


def _squash(s: str) -> str:
    return _norm(s).replace(" ", "")


def _match_field(raw_key: str) -> tuple[str | None, float]:
    """Map a label found on the form to a canonical field name + match confidence."""
    key = _norm(raw_key)
    if not key:
        return None, 0.0

    best_field, best_score = None, 0.0
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            alias_n = _norm(alias)
            if key == alias_n:
                return canonical, 1.0
            if _squash(key) == _squash(alias_n):
                return canonical, 0.98
            # Substring hit, e.g. "how often do you exercise (per week)"
            if alias_n in key or key in alias_n:
                score = 0.92
            else:
                score = difflib.SequenceMatcher(None, key, alias_n).ratio()
            if score > best_score:
                best_field, best_score = canonical, score

    if best_score >= KEY_MATCH_CUTOFF:
        return best_field, round(best_score, 4)
    return None, round(best_score, 4)


def _fuzzy_in(value: str, vocabulary) -> tuple[bool, float]:
    """
    True if `value` matches any vocabulary entry, with a confidence.

    Matching is by whole word, never bare substring: a substring test lets the
    single-letter token "n" match inside "occasionally" and silently flip a
    smoker to a non-smoker.
    """
    v = _norm(value)
    if not v:
        return False, 0.0

    for word in vocabulary:
        if v == _norm(word):
            return True, 1.0

    for word in vocabulary:
        w = _norm(word)
        if w and re.search(rf"(?<!\w){re.escape(w)}(?!\w)", v):
            return True, 0.93

    for word in vocabulary:
        ratio = difflib.SequenceMatcher(None, v, _norm(word)).ratio()
        if ratio >= 0.85:
            return True, round(ratio * 0.95, 4)

    return False, 0.0


# --------------------------------------------------------------------------
# value normalisers: each returns (value, confidence) or (None, 0.0)
# --------------------------------------------------------------------------

def _parse_age(value: Any) -> tuple[int | None, float]:
    if isinstance(value, bool):
        return None, 0.0
    if isinstance(value, (int, float)):
        age = int(value)
        return (age, 1.0) if 0 < age <= 120 else (None, 0.0)
    text = str(value).lower()
    match = re.search(r"\d{1,3}", text)
    if not match:
        # OCR turns 42 into "4Z" or 50 into "SO". Only attempt that repair when
        # the token is plausibly numeric - otherwise "abc" becomes 6 via b->6.
        token = re.sub(r"[^a-z0-9]", "", text)
        confusable = set("oliszbg")
        if token and len(token) <= 3 and set(token) <= confusable | set("0123456789"):
            repaired = (token.replace("o", "0").replace("l", "1").replace("i", "1")
                             .replace("z", "2").replace("s", "5")
                             .replace("b", "6").replace("g", "9"))
            match = re.search(r"\d{1,3}", repaired)
    if not match:
        return None, 0.0
    age = int(match.group())
    if not 0 < age <= 120:
        return None, 0.0
    return age, 1.0 if match.group() in text else 0.80


def _parse_bool(value: Any) -> tuple[bool | None, float]:
    if isinstance(value, bool):
        return value, 1.0
    if isinstance(value, (int, float)):
        return bool(value), 0.95
    text = _norm(value)
    # Check negatives first: "non-smoker" contains "smoker", a true-word.
    hit, conf = _fuzzy_in(text, FALSE_WORDS)
    if hit:
        return False, min(conf, 0.99)
    hit, conf = _fuzzy_in(text, TRUE_WORDS)
    if hit:
        return True, min(conf, 0.99)
    return None, 0.0


def _parse_smoker(value: Any) -> tuple[Any, float]:
    """Keeps the raw qualifier ('occasionally') so scoring can be graded."""
    parsed, conf = _parse_bool(value)
    if parsed is None:
        return None, 0.0
    if parsed and isinstance(value, str):
        return {"value": True, "detail": _norm(value)}, conf
    return parsed, conf


def _parse_exercise(value: Any) -> tuple[str | None, float]:
    text = _norm(value)
    if text in EXERCISE_LEVELS:
        return text, 1.0
    best, best_ratio = None, 0.0
    for label in EXERCISE_LEVELS:
        if label in text or text in label:
            return label, 0.93
        ratio = difflib.SequenceMatcher(None, text, label).ratio()
        if ratio > best_ratio:
            best, best_ratio = label, ratio
    if best_ratio >= 0.80:
        return best, round(best_ratio * 0.95, 4)
    # Numeric answers: "3 times a week"
    match = re.search(r"(\d+)\s*(?:x|times|days|sessions)?", text)
    if match:
        n = int(match.group(1))
        label = "never" if n == 0 else "rarely" if n == 1 else "sometimes" if n == 2 \
            else "often" if n <= 4 else "daily"
        return label, 0.85
    return None, 0.0


def _parse_diet(value: Any) -> tuple[str | None, float]:
    text = _norm(value)
    if not text:
        return None, 0.0
    poor = [k for k in DIET_POOR_KEYWORDS if _norm(k) in text]
    good = [k for k in DIET_GOOD_KEYWORDS if _norm(k) in text]
    if poor or good:
        # Keep the original wording - the rationale reads better with it.
        return text, 0.95 if (poor and not good) or (good and not poor) else 0.80
    return text, 0.62      # recorded, but we are not confident about its class


def _parse_alcohol(value: Any) -> tuple[str | None, float]:
    text = _norm(value)
    if not text:
        return None, 0.0
    if any(_norm(k) in text for k in ALCOHOL_HEAVY_KEYWORDS):
        return text, 0.93
    if any(_norm(k) in text for k in ALCOHOL_LIGHT_KEYWORDS):
        return text, 0.93
    return text, 0.65


def _parse_sleep(value: Any) -> tuple[float | None, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        hours = float(value)
        return (hours, 1.0) if 0 < hours <= 24 else (None, 0.0)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None, 0.0
    hours = float(match.group())
    return (hours, 0.9) if 0 < hours <= 24 else (None, 0.0)


def _parse_bmi(value: Any) -> tuple[float | None, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        bmi = float(value)
        return (bmi, 1.0) if 8 < bmi < 90 else (None, 0.0)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None, 0.0
    bmi = float(match.group())
    return (bmi, 0.9) if 8 < bmi < 90 else (None, 0.0)


def _parse_family_history(value: Any) -> tuple[Any, float]:
    parsed, conf = _parse_bool(value)
    if parsed is not None:
        return parsed, conf
    text = _norm(value)
    if any(k in text for k in FAMILY_HISTORY_POSITIVE):
        return text, 0.9
    return (text, 0.6) if text else (None, 0.0)


NORMALISERS = {
    "age": _parse_age,
    "smoker": _parse_smoker,
    "exercise": _parse_exercise,
    "diet": _parse_diet,
    "alcohol": _parse_alcohol,
    "sleep_hours": _parse_sleep,
    "bmi": _parse_bmi,
    "family_history": _parse_family_history,
}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def _pairs_from_text(text: str) -> list[tuple[str, str]]:
    """Pull (label, value) pairs out of free-form / OCR'd form text."""
    pairs: list[tuple[str, str]] = []
    for line in _LINE_SPLIT.split(text):
        line = line.strip().strip("|•*").strip()
        if not line:
            continue
        # A line may hold several fields: "Age: 42   Smoker: yes"
        segments = re.split(r"\s{2,}(?=[A-Za-z][\w ]{2,20}\s*[:=])", line)
        for segment in segments:
            parts = _KV_SPLIT.split(segment, maxsplit=1)
            if len(parts) == 2 and parts[0].strip():
                pairs.append((parts[0], parts[1]))
            else:
                # No delimiter: "Smoker yes" / "Age 42"
                tokens = segment.split()
                for split_at in range(min(4, len(tokens) - 1), 0, -1):
                    candidate, _score = _match_field(" ".join(tokens[:split_at]))
                    if candidate:
                        pairs.append((" ".join(tokens[:split_at]),
                                      " ".join(tokens[split_at:])))
                        break
    return pairs


def _pairs_from_obj(obj: dict) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for key, value in obj.items():
        if isinstance(value, dict):
            pairs.extend(_pairs_from_obj(value))     # tolerate nested payloads
        else:
            pairs.append((key, value))
    return pairs


def parse_survey(payload: Any, source: str = "json",
                 ocr_confidence: float | None = None) -> ParseResult:
    """
    Normalise any accepted input shape into a ParseResult.

    payload : dict (typed JSON) or str (raw / OCR'd text)
    source  : 'json' | 'text' | 'image'
    ocr_confidence : mean OCR word confidence, when source == 'image'
    """
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                pass

    if isinstance(payload, dict):
        raw_pairs: list[tuple[str, Any]] = _pairs_from_obj(payload)
    else:
        raw_pairs = list(_pairs_from_text(str(payload)))

    answers: dict[str, Any] = {}
    field_conf: dict[str, float] = {}
    unrecognised: list[str] = []

    for raw_key, raw_value in raw_pairs:
        canonical, key_conf = _match_field(str(raw_key))
        if canonical is None:
            unrecognised.append(str(raw_key).strip())
            continue
        if isinstance(raw_value, str) and _squash(raw_value) in _NULLISH_SQUASHED:
            continue
        if raw_value is None:
            continue

        value, value_conf = NORMALISERS[canonical](raw_value)
        if value is None:
            unrecognised.append(f"{raw_key}={raw_value}")
            continue

        combined = round(key_conf * value_conf, 4)
        # Keep the better reading if a field appears twice on the form.
        if canonical in answers and field_conf.get(canonical, 0) >= combined:
            continue
        answers[canonical] = value
        field_conf[canonical] = combined

    missing = [f for f in REQUIRED_FIELDS if f not in answers]

    confidence = _aggregate_confidence(field_conf, missing, source, ocr_confidence)

    # Present `smoker` in its simple boolean form; the qualifier moves aside so
    # the output matches the published contract exactly.
    qualifiers: dict[str, str] = {}
    public_answers = dict(answers)
    if isinstance(public_answers.get("smoker"), dict):
        public_answers["smoker"] = answers["smoker"]["value"]
        qualifiers["smoker"] = answers["smoker"]["detail"]

    ordered = {k: public_answers[k] for k in ALL_FIELDS if k in public_answers}
    ordered.update({k: v for k, v in public_answers.items() if k not in ordered})

    return ParseResult(
        answers=ordered,
        missing_fields=missing,
        confidence=confidence,
        field_confidence=field_conf,
        source=source,
        unrecognised=unrecognised,
        qualifiers=qualifiers,
    )


def _aggregate_confidence(field_conf: dict[str, float], missing: list[str],
                          source: str, ocr_confidence: float | None) -> float:
    """Mean field confidence, penalised by channel noise and by coverage gaps."""
    if not field_conf:
        return 0.0

    mean = sum(field_conf.values()) / len(field_conf)
    channel = SOURCE_CONFIDENCE.get(source, 0.95)

    if source == "image" and ocr_confidence is not None:
        # Blend the declared channel prior with what Tesseract actually reported.
        channel = round(0.5 * channel + 0.5 * max(ocr_confidence, 0.5), 4)

    coverage = 1 - (len(missing) / len(REQUIRED_FIELDS))
    coverage_penalty = 1.0 if coverage == 1 else 0.85 + 0.15 * coverage

    return round(min(mean * channel * coverage_penalty, 0.99), 2)


# ==========================================================================
# STEPS 2-4 - FACTORS, RISK SCORE, RECOMMENDATIONS
# Rule-based on purpose: a screening score has to be explainable. Every point
# traces back to a named factor; every recommendation to a factor that was
# actually present.
# ==========================================================================


# --------------------------------------------------------------------------
# Step 2 - factor extraction
# --------------------------------------------------------------------------

@dataclass
class FactorResult:
    factors: list[str]
    confidence: float
    age_factor: str | None = None
    details: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"factors": self.factors, "confidence": self.confidence}


def _age_factor(age: int | None) -> str | None:
    if age is None:
        return None
    if age < 30:
        return None
    if age < 40:
        return "age 30-39"
    if age < 50:
        return "age 40-49"
    if age < 60:
        return "age 50-59"
    if age < 70:
        return "age 60-69"
    return "age 70+"


def extract_factors(answers: dict[str, Any], parse_confidence: float = 0.9,
                    field_confidence: dict[str, float] | None = None,
                    qualifiers: dict[str, str] | None = None) -> FactorResult:
    factors: list[str] = []
    details: dict[str, str] = {}
    field_confidence = field_confidence or {}
    qualifiers = qualifiers or {}
    used_fields: list[str] = []

    # --- smoking -----------------------------------------------------------
    smoker = answers.get("smoker")
    detail = str(qualifiers.get("smoker", "")).lower()
    if smoker is True:
        used_fields.append("smoker")
        if any(word in detail for word in SMOKER_LIGHT_WORDS):
            factors.append("occasional smoking")
            details["occasional smoking"] = detail or "occasional"
        else:
            factors.append("smoking")
            details["smoking"] = detail or "current smoker"

    # --- diet --------------------------------------------------------------
    diet = str(answers.get("diet", "")).lower()
    if diet:
        used_fields.append("diet")
        poor_hits = [k for k in DIET_POOR_KEYWORDS if k in diet]
        good_hits = [k for k in DIET_GOOD_KEYWORDS if k in diet]
        if poor_hits and len(poor_hits) >= len(good_hits):
            factors.append("poor diet")
            details["poor diet"] = f"{diet} ({', '.join(poor_hits[:3])})"

    # --- exercise ----------------------------------------------------------
    exercise = str(answers.get("exercise", "")).lower()
    if exercise:
        used_fields.append("exercise")
        level = EXERCISE_LEVELS.get(exercise)
        if level == 0:
            factors.append("sedentary lifestyle")
            details["sedentary lifestyle"] = exercise
        elif level is not None and level <= 1:
            factors.append("low exercise")
            details["low exercise"] = exercise

    # --- alcohol -----------------------------------------------------------
    alcohol = str(answers.get("alcohol", "")).lower()
    if alcohol:
        used_fields.append("alcohol")
        if any(k in alcohol for k in ALCOHOL_HEAVY_KEYWORDS):
            factors.append("heavy alcohol use")
            details["heavy alcohol use"] = alcohol

    # --- sleep -------------------------------------------------------------
    sleep = answers.get("sleep_hours")
    if isinstance(sleep, (int, float)):
        used_fields.append("sleep_hours")
        if sleep < 6:
            factors.append("insufficient sleep")
            details["insufficient sleep"] = f"{sleep} h/night"

    # --- BMI ---------------------------------------------------------------
    bmi = answers.get("bmi")
    if isinstance(bmi, (int, float)):
        used_fields.append("bmi")
        if bmi >= 30:
            factors.append("obesity")
            details["obesity"] = f"BMI {bmi}"
        elif bmi >= 25:
            factors.append("overweight")
            details["overweight"] = f"BMI {bmi}"

    # --- family history ----------------------------------------------------
    history = answers.get("family_history")
    if history not in (None, False, ""):
        used_fields.append("family_history")
        factors.append("family history")
        details["family history"] = str(history)

    # --- age (scored, but reported separately from behavioural factors) -----
    age_factor = _age_factor(answers.get("age"))
    if age_factor:
        used_fields.append("age")

    # Confidence: the mean confidence of the fields that actually produced a
    # verdict, discounted slightly because classification adds its own error.
    contributing = [field_confidence[f] for f in used_fields if f in field_confidence]
    if contributing:
        # Weight the parse confidence by how clean the specific fields that drove
        # the verdict were, rather than by the form as a whole.
        field_mean = sum(contributing) / len(contributing)
        confidence = round(min(parse_confidence * field_mean * 0.98, 0.99), 2)
    else:
        confidence = round(parse_confidence * 0.96, 2)

    return FactorResult(factors=factors, confidence=confidence,
                        age_factor=age_factor, details=details)


# --------------------------------------------------------------------------
# Step 3 - risk classification
# --------------------------------------------------------------------------

@dataclass
class RiskResult:
    risk_level: str
    score: int
    rationale: list[str]
    breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"risk_level": self.risk_level, "score": self.score,
                "rationale": self.rationale}


def _band(score: int) -> str:
    for low, high, level in RISK_BANDS:
        if low <= score <= high:
            return level
    return "high"


def classify_risk(factors: list[str], age_factor: str | None = None) -> RiskResult:
    breakdown: dict[str, int] = {}
    for factor in factors:
        weight = FACTOR_WEIGHTS.get(factor)
        if weight:
            breakdown[factor] = weight
    if age_factor:
        breakdown[age_factor] = FACTOR_WEIGHTS.get(age_factor, 0)

    score = min(sum(breakdown.values()), 100)
    rationale = [FACTOR_RATIONALE.get(f, f) for f in factors]

    return RiskResult(risk_level=_band(score), score=score,
                      rationale=rationale, breakdown=breakdown)


# --------------------------------------------------------------------------
# Step 4 - recommendations
# --------------------------------------------------------------------------

def build_recommendations(factors: list[str], risk_level: str,
                          age_factor: str | None = None) -> list[str]:
    """Highest-impact factor first, deduplicated, capped."""
    ordered = sorted(factors, key=lambda f: FACTOR_WEIGHTS.get(f, 0), reverse=True)

    primary: list[str] = []      # one headline action per factor
    secondary: list[str] = []    # supporting detail, used only if room remains

    for factor in ordered:
        advice = RECOMMENDATIONS.get(factor, [])
        if advice:
            primary.append(advice[0])
            secondary.extend(advice[1:])

    if age_factor and age_factor in AGE_RECOMMENDATIONS:
        secondary.append(AGE_RECOMMENDATIONS[age_factor])

    if not primary:
        primary = list(BASELINE_RECOMMENDATIONS)

    out: list[str] = []
    for item in primary + secondary:
        if item not in out:
            out.append(item)
        if len(out) >= MAX_RECOMMENDATIONS:
            break
    return out


@dataclass
class Profile:
    risk_level: str
    factors: list[str]
    recommendations: list[str]
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "factors": self.factors,
            "recommendations": self.recommendations,
            "status": self.status,
        }


# ==========================================================================
# ORCHESTRATION + GUARDRAILS
# ==========================================================================


class GuardrailExit(Exception):
    """Raised when the profile cannot be produced responsibly."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload.get("reason", "guardrail triggered"))
        self.payload = payload


def check_guardrails(parsed: ParseResult) -> None:
    missing_ratio = len(parsed.missing_fields) / len(REQUIRED_FIELDS)

    if missing_ratio > MISSING_FIELD_THRESHOLD:
        raise GuardrailExit({
            "status": "incomplete_profile",
            "reason": ">50% fields missing",
            "missing_fields": parsed.missing_fields,
            "confidence": parsed.confidence,
        })

    if parsed.confidence < MIN_CONFIDENCE:
        raise GuardrailExit({
            "status": "incomplete_profile",
            "reason": "input too noisy to parse reliably",
            "missing_fields": parsed.missing_fields,
            "confidence": parsed.confidence,
        })


def ingest(payload: Any = None, image_bytes: bytes | None = None) -> ParseResult:
    """Step 1 for any input channel."""
    if image_bytes is not None:
        ocr = image_to_text(image_bytes)
        return parse_survey(ocr.text, source="image", ocr_confidence=ocr.confidence)
    source = "json" if isinstance(payload, dict) else "text"
    return parse_survey(payload, source=source)


def run_pipeline(payload: Any = None, image_bytes: bytes | None = None,
                 verbose: bool = False) -> dict[str, Any]:
    """
    Full steps 1-4. Returns the step 4 contract, optionally with the
    intermediate step outputs attached for demo / debugging.
    """
    parsed = ingest(payload=payload, image_bytes=image_bytes)
    check_guardrails(parsed)                      # may raise GuardrailExit

    factors = extract_factors(
        parsed.answers,
        parse_confidence=parsed.confidence,
        field_confidence=parsed.field_confidence,
        qualifiers=parsed.qualifiers,
    )
    risk = classify_risk(factors.factors, age_factor=factors.age_factor)
    recommendations = build_recommendations(
        factors.factors, risk.risk_level, age_factor=factors.age_factor
    )

    profile = Profile(
        risk_level=risk.risk_level,
        factors=factors.factors,
        recommendations=recommendations,
        status="ok",
    )

    result = profile.to_dict()
    result["disclaimer"] = DISCLAIMER

    if verbose:
        result["steps"] = {
            "1_parse": parsed.to_dict(),
            "2_factors": factors.to_dict(),
            "3_risk": risk.to_dict(),
        }
        result["debug"] = {
            "score": risk.score,
            "score_breakdown": risk.breakdown,
            "age_factor": factors.age_factor,
            "field_confidence": parsed.field_confidence,
            "unrecognised_input": parsed.unrecognised,
            "source": parsed.source,
        }
    return result


# ==========================================================================
# HTTP SURFACE
# Every step is individually addressable so each stage can be tested and demoed
# on its own, plus one endpoint that runs the whole thing.
# ==========================================================================

# FastAPI is optional. Without it the module still imports, and the parsing,
# scoring and CLI paths all work - only the HTTP routes are unavailable.
if FASTAPI_AVAILABLE:

    app = FastAPI(
        title="AI-Powered Health Risk Profiler",
        version="1.0.0",
        description="OCR -> factor extraction -> risk scoring -> recommendations. "
                    "Non-diagnostic lifestyle screening.",
    )


    # --------------------------------------------------------------------------
    # request models
    # --------------------------------------------------------------------------

    class TextInput(BaseModel):
        text: str = Field(..., description="Raw or OCR'd survey text")


    class FactorsInput(BaseModel):
        answers: dict[str, Any]
        confidence: float = 0.9


    class RiskInput(BaseModel):
        factors: list[str]
        age_factor: str | None = None


    class RecommendInput(BaseModel):
        factors: list[str]
        risk_level: str = "moderate"
        age_factor: str | None = None


    # --------------------------------------------------------------------------
    # meta
    # --------------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "ocr_available": OCR_AVAILABLE}


    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/demo")


    @app.get("/demo", include_in_schema=False)
    def demo() -> HTMLResponse:
        """Browser client that calls the same public endpoints as any other caller."""
        return HTMLResponse(DEMO_PAGE)


    # --------------------------------------------------------------------------
    # Step 1 - OCR / text parsing
    # --------------------------------------------------------------------------

    @app.post("/v1/parse")
    def parse_endpoint(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        """
        Accepts either the survey object itself, or {"text": "..."} for raw text.
        Returns answers / missing_fields / confidence, or the guardrail exit object.
        """
        body = payload.get("text") if set(payload) == {"text"} else payload
        parsed = ingest(payload=body)
        try:
            check_guardrails(parsed)
        except GuardrailExit as exit_:
            return JSONResponse(exit_.payload, status_code=422)
        return JSONResponse(parsed.to_dict())


    @app.post("/v1/parse/image")
    async def parse_image_endpoint(file: UploadFile = File(...)) -> JSONResponse:
        try:
            parsed = ingest(image_bytes=await file.read())
        except OCRUnavailable as exc:
            return JSONResponse({"status": "ocr_unavailable", "reason": str(exc)},
                                status_code=503)
        try:
            check_guardrails(parsed)
        except GuardrailExit as exit_:
            return JSONResponse(exit_.payload, status_code=422)
        return JSONResponse(parsed.to_dict())


    # --------------------------------------------------------------------------
    # Step 2 - factor extraction
    # --------------------------------------------------------------------------

    @app.post("/v1/factors")
    def factors_endpoint(payload: FactorsInput) -> JSONResponse:
        result = extract_factors(payload.answers, parse_confidence=payload.confidence)
        return JSONResponse(result.to_dict())


    # --------------------------------------------------------------------------
    # Step 3 - risk classification
    # --------------------------------------------------------------------------

    @app.post("/v1/risk")
    def risk_endpoint(payload: RiskInput) -> JSONResponse:
        result = classify_risk(payload.factors, age_factor=payload.age_factor)
        return JSONResponse(result.to_dict())


    # --------------------------------------------------------------------------
    # Step 4 - recommendations
    # --------------------------------------------------------------------------

    @app.post("/v1/recommendations")
    def recommendations_endpoint(payload: RecommendInput) -> JSONResponse:
        recs = build_recommendations(payload.factors, payload.risk_level,
                                     age_factor=payload.age_factor)
        return JSONResponse({
            "risk_level": payload.risk_level,
            "factors": payload.factors,
            "recommendations": recs,
            "status": "ok",
            "disclaimer": DISCLAIMER,
        })


    # --------------------------------------------------------------------------
    # Full pipeline
    # --------------------------------------------------------------------------

    @app.post("/v1/profile")
    def profile_endpoint(payload: dict[str, Any] = Body(...),
                         verbose: bool = False) -> JSONResponse:
        body = payload.get("text") if set(payload) == {"text"} else payload
        try:
            return JSONResponse(run_pipeline(payload=body, verbose=verbose))
        except GuardrailExit as exit_:
            return JSONResponse(exit_.payload, status_code=422)


    @app.post("/v1/profile/image")
    async def profile_image_endpoint(file: UploadFile = File(...),
                                     verbose: bool = Form(False)) -> JSONResponse:
        try:
            return JSONResponse(run_pipeline(image_bytes=await file.read(),
                                             verbose=verbose))
        except OCRUnavailable as exc:
            return JSONResponse({"status": "ocr_unavailable", "reason": str(exc)},
                                status_code=503)
        except GuardrailExit as exit_:
            return JSONResponse(exit_.payload, status_code=422)


# ==========================================================================
# DEMO PAGE
# Browser client for the API. Served at /demo; it calls the same public
# endpoints any other caller would.
# ==========================================================================

DEMO_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Health Risk Profiler</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#12262b; --ink-soft:#4a6068; --line:#d3dedd; --rule:#e8eeed;
    --paper:#ffffff; --field:#f2f6f5; --wash:#e9f0ef;
    --teal:#0f6e6e;
    --low:#2e7d5b; --moderate:#b8721b; --high:#b33a3a;
    --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
    --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    font-family:var(--sans); color:var(--ink); background:var(--wash);
    font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
  }
  .shell{max-width:1120px;margin:0 auto;padding:40px 24px 72px}

  header{margin-bottom:28px;max-width:62ch}
  h1{font-size:clamp(28px,4vw,38px);font-weight:700;letter-spacing:-.022em;margin:0 0 6px;line-height:1.1}
  header p{margin:0;color:var(--ink-soft)}
  .nondx{
    display:inline-block;margin-top:14px;padding:5px 11px;border-radius:2px;
    background:var(--paper);border-left:3px solid var(--teal);
    font-size:13px;color:var(--ink-soft);
  }

  .grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,4fr);gap:24px;align-items:start}
  @media (max-width:880px){.grid{grid-template-columns:1fr}}

  .panel{background:var(--paper);border:1px solid var(--line);border-radius:4px;padding:20px}
  .panel h2{font-size:15px;font-weight:600;margin:0 0 14px;letter-spacing:-.01em}

  .presets{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
  .presets button{
    font:inherit;font-size:13px;padding:5px 10px;border-radius:2px;cursor:pointer;
    background:var(--field);color:var(--ink-soft);border:1px solid var(--line);
  }
  .presets button:hover{color:var(--ink);border-color:var(--ink-soft)}

  textarea{
    width:100%;min-height:172px;resize:vertical;padding:12px;
    font-family:var(--mono);font-size:13.5px;line-height:1.6;color:var(--ink);
    background:var(--field);border:1px solid var(--line);border-radius:3px;
  }
  textarea:focus,button:focus-visible,label.file:focus-within{outline:2px solid var(--teal);outline-offset:2px}

  .actions{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap}
  .run{
    font:inherit;font-weight:600;font-size:14.5px;padding:10px 20px;cursor:pointer;
    background:var(--ink);color:#fff;border:0;border-radius:3px;
  }
  .run:hover{background:var(--teal)}
  .run[disabled]{opacity:.5;cursor:progress}
  label.file{font-size:13.5px;color:var(--teal);cursor:pointer;border-bottom:1px solid currentColor}
  label.file input{position:absolute;width:1px;height:1px;opacity:0}
  .filename{font-size:13px;color:var(--ink-soft);font-family:var(--mono)}

  /* ---- pipeline stages ---- */
  .stage{border-top:1px solid var(--rule);padding:16px 0}
  .stage:first-of-type{border-top:0;padding-top:0}
  .stage-head{display:flex;align-items:baseline;gap:10px}
  .stage-n{font-family:var(--mono);font-size:12px;color:var(--teal);width:16px;flex:none}
  .stage-title{font-weight:600;font-size:14px}
  .stage-conf{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--ink-soft)}
  .stage-body{margin:10px 0 0 26px}
  .stage.idle{opacity:.38}
  .stage.idle .stage-body{display:none}

  pre{
    margin:0;padding:11px 12px;background:var(--field);border-radius:3px;
    font-family:var(--mono);font-size:12.5px;line-height:1.6;overflow-x:auto;
    color:var(--ink);white-space:pre;
  }
  .chips{display:flex;flex-wrap:wrap;gap:6px}
  .chip{
    font-size:13px;padding:4px 10px;border-radius:2px;
    background:var(--field);border:1px solid var(--line);
  }
  .chip.none{color:var(--ink-soft);font-style:italic}

  /* ---- the score meter: segments sized by each factor's points ---- */
  .meter{margin-top:4px}
  .meter-score{display:flex;align-items:baseline;gap:9px;margin-bottom:9px}
  .meter-score b{font-size:34px;font-weight:700;letter-spacing:-.03em;line-height:1}
  .meter-level{font-weight:600;font-size:14px}
  .meter-of{font-family:var(--mono);font-size:12px;color:var(--ink-soft)}
  .bar{display:flex;height:26px;background:var(--field);border-radius:2px;overflow:hidden}
  .seg{
    display:flex;align-items:center;justify-content:center;min-width:0;
    font-family:var(--mono);font-size:11px;color:#fff;
    width:0;transition:width .55s cubic-bezier(.2,.7,.3,1);
  }
  .ticks{position:relative;height:15px;margin-top:3px}
  .tick{position:absolute;top:0;font-family:var(--mono);font-size:10.5px;color:var(--ink-soft);transform:translateX(-50%)}
  .tick::before{content:"";position:absolute;top:-3px;left:50%;width:1px;height:4px;background:var(--line)}
  .legend{margin:11px 0 0;padding:0;list-style:none;display:grid;gap:5px}
  .legend li{display:flex;align-items:center;gap:8px;font-size:13.5px}
  .swatch{width:9px;height:9px;border-radius:1px;flex:none}
  .legend .pts{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--ink-soft)}

  ol.recs{margin:0;padding-left:20px}
  ol.recs li{margin-bottom:5px}
  ol.recs li::marker{color:var(--teal);font-weight:600}

  .blocked{border-left:3px solid var(--high);background:#fbf3f3;padding:13px 15px;border-radius:0 3px 3px 0}
  .blocked h3{margin:0 0 4px;font-size:14px;font-weight:600;color:var(--high)}
  .blocked p{margin:0;font-size:13.5px;color:var(--ink-soft)}
  footer{margin-top:22px;font-size:12.5px;color:var(--ink-soft);max-width:70ch}
  @media (prefers-reduced-motion:reduce){.seg{transition:none}}
</style>
</head>
<body>
<div class="shell">
  <header>
    <h1>Health risk profiler</h1>
    <p>Paste a lifestyle survey or upload a scanned form. The service parses it, pulls out risk factors, scores them, and returns guidance.</p>
    <div class="nondx">Screening heuristic only. Not a diagnosis.</div>
  </header>

  <div class="grid">
    <section class="panel">
      <h2>Survey input</h2>
      <div class="presets" id="presets"></div>
      <textarea id="input" spellcheck="false" aria-label="Survey input"></textarea>
      <div class="actions">
        <button class="run" id="run">Build profile</button>
        <label class="file">Upload a scan<input type="file" id="file" accept="image/*"></label>
        <span class="filename" id="filename"></span>
      </div>
    </section>

    <section class="panel" id="output">
      <h2>Pipeline</h2>

      <div class="stage idle" id="s1">
        <div class="stage-head"><span class="stage-n">1</span><span class="stage-title">Parsed answers</span><span class="stage-conf" id="c1"></span></div>
        <div class="stage-body"><pre id="b1"></pre></div>
      </div>

      <div class="stage idle" id="s2">
        <div class="stage-head"><span class="stage-n">2</span><span class="stage-title">Risk factors</span><span class="stage-conf" id="c2"></span></div>
        <div class="stage-body"><div class="chips" id="b2"></div></div>
      </div>

      <div class="stage idle" id="s3">
        <div class="stage-head"><span class="stage-n">3</span><span class="stage-title">Score</span></div>
        <div class="stage-body">
          <div class="meter">
            <div class="meter-score"><b id="score">0</b><span class="meter-of">/ 100</span><span class="meter-level" id="level"></span></div>
            <div class="bar" id="bar"></div>
            <div class="ticks"><span class="tick" style="left:25%">25</span><span class="tick" style="left:55%">55</span></div>
            <ul class="legend" id="legend"></ul>
          </div>
        </div>
      </div>

      <div class="stage idle" id="s4">
        <div class="stage-head"><span class="stage-n">4</span><span class="stage-title">Recommendations</span></div>
        <div class="stage-body"><ol class="recs" id="b4"></ol></div>
      </div>

      <div id="blocked"></div>
    </section>
  </div>

  <footer id="disclaimer"></footer>
</div>

<script>
const PRESETS = {
  "Reference case": '{"age":42,"smoker":true,"exercise":"rarely","diet":"high sugar"}',
  "Scanned form text": "Age: 42\nSmoker: yes\nExercise: rarely\nDiet: high sugar",
  "Noisy OCR": "Aqe: 42\nSrnoker: yes\nExercse: rarely\nDlet: high sugar",
  "Low risk": '{"age":26,"smoker":false,"exercise":"daily","diet":"balanced home cooked"}',
  "Extra fields": '{"age":58,"smoker":"occasionally","exercise":"never","diet":"fast food","alcohol":"daily","sleep_hours":5,"bmi":31}',
  "Too incomplete": '{"age":42,"smoker":"N/A"}'
};

const $ = id => document.getElementById(id);
const SEG_COLORS = ["#12262b","#0f6e6e","#3f8f7c","#7aa88f","#b8721b","#b33a3a","#6b7f85"];

const presetBar = $("presets");
Object.keys(PRESETS).forEach((name, i) => {
  const b = document.createElement("button");
  b.textContent = name;
  b.onclick = () => { $("input").value = PRESETS[name]; $("file").value = ""; $("filename").textContent = ""; run(); };
  presetBar.appendChild(b);
  if (i === 0) $("input").value = PRESETS[name];
});

$("file").onchange = e => {
  const f = e.target.files[0];
  $("filename").textContent = f ? f.name : "";
  if (f) run();
};

function reset() {
  ["s1","s2","s3","s4"].forEach(id => $(id).classList.add("idle"));
  $("blocked").innerHTML = "";
  $("bar").innerHTML = ""; $("legend").innerHTML = "";
  $("disclaimer").textContent = "";
}

async function run() {
  reset();
  $("run").disabled = true;
  try {
    const file = $("file").files[0];
    let res;
    if (file) {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("verbose", "true");
      res = await fetch("/v1/profile/image", { method: "POST", body: fd });
    } else {
      const raw = $("input").value.trim();
      let body;
      try { body = JSON.parse(raw); } catch { body = { text: raw }; }
      res = await fetch("/v1/profile?verbose=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
    }
    render(await res.json(), res.ok);
  } catch (err) {
    showBlocked("Request failed", String(err));
  } finally {
    $("run").disabled = false;
  }
}

function showBlocked(title, reason) {
  $("blocked").innerHTML =
    `<div class="blocked"><h3>${title}</h3><p>${reason}</p></div>`;
}

function render(data, ok) {
  if (!ok) {
    if (data.missing_fields) {
      $("s1").classList.remove("idle");
      $("c1").textContent = "confidence " + (data.confidence ?? 0);
      $("b1").textContent = JSON.stringify({ missing_fields: data.missing_fields }, null, 2);
    }
    showBlocked(
      data.status === "ocr_unavailable" ? "OCR unavailable" : "Profile not generated",
      data.reason + " — the service stops here rather than guessing the rest."
    );
    return;
  }

  const steps = data.steps || {}, dbg = data.debug || {};

  // 1 - parsed answers
  $("s1").classList.remove("idle");
  $("c1").textContent = "confidence " + steps["1_parse"].confidence;
  $("b1").textContent = JSON.stringify(steps["1_parse"].answers, null, 2);

  // 2 - factors
  $("s2").classList.remove("idle");
  $("c2").textContent = "confidence " + steps["2_factors"].confidence;
  $("b2").innerHTML = data.factors.length
    ? data.factors.map(f => `<span class="chip">${f}</span>`).join("")
    : '<span class="chip none">no risk factors found</span>';

  // 3 - score, built visibly out of its parts
  $("s3").classList.remove("idle");
  const score = dbg.score ?? 0;
  $("score").textContent = score;
  const level = steps["3_risk"].risk_level;
  $("level").textContent = level;
  $("level").style.color = `var(--${level})`;

  const breakdown = Object.entries(dbg.score_breakdown || {});
  const bar = $("bar"), legend = $("legend");
  breakdown.forEach(([name, pts], i) => {
    const color = SEG_COLORS[i % SEG_COLORS.length];
    const seg = document.createElement("div");
    seg.className = "seg";
    seg.style.background = color;
    seg.title = `${name}: ${pts} points`;
    if (pts >= 12) seg.textContent = pts;
    bar.appendChild(seg);
    requestAnimationFrame(() => { seg.style.width = pts + "%"; });

    legend.insertAdjacentHTML("beforeend",
      `<li><span class="swatch" style="background:${color}"></span>${name}<span class="pts">+${pts}</span></li>`);
  });
  if (!breakdown.length) {
    legend.innerHTML = '<li style="color:var(--ink-soft)">Nothing scored — no risk factors present.</li>';
  }

  // 4 - recommendations
  $("s4").classList.remove("idle");
  $("b4").innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join("");
  $("disclaimer").textContent = data.disclaimer || "";
}

$("run").onclick = run;
run();
</script>
</body>
</html>
"""


# ==========================================================================
# SAMPLE FORM GENERATOR
# Renders a scan-like survey image so the OCR path can be demoed without a
# scanner. Slight rotation and speckle make it a fair test rather than a
# clean synthetic render.
# ==========================================================================

SAMPLE_FORM_LINES = (
    ("LIFESTYLE HEALTH SURVEY", True),
    ("", False),
    ("Age: 42", False),
    ("Smoker: yes", False),
    ("Exercise: rarely", False),
    ("Diet: high sugar", False),
)


def _sample_font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    )
    from PIL import ImageFont
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sample_form(path: str = "form.png", noisy: bool = True) -> str:
    """Write a scan-like survey image to `path` and return the path."""
    if not OCR_AVAILABLE:
        raise OCRUnavailable("Pillow is required to render the sample form.")

    import random

    from PIL import ImageDraw

    width, height = 900, 520
    image = Image.new("RGB", (width, height), (252, 250, 245))
    draw = ImageDraw.Draw(image)

    y = 60
    for text, is_title in SAMPLE_FORM_LINES:
        if text:
            draw.text((70, y), text,
                      font=_sample_font(34 if is_title else 30, is_title),
                      fill=(25, 25, 30))
        y += 70 if is_title else 60

    if noisy:
        image = image.rotate(-0.6, expand=False, fillcolor=(252, 250, 245))
        image = image.filter(ImageFilter.GaussianBlur(0.4))
        pixels = image.load()
        for _ in range(int(width * height * 0.015)):
            x, yy = random.randint(0, width - 1), random.randint(0, height - 1)
            shade = random.randint(150, 235)
            pixels[x, yy] = (shade, shade, shade)

    image.save(path)
    return path


# ==========================================================================
# COMMAND-LINE ENTRY POINT
# ==========================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="health-risk-profiler",
        description="Non-diagnostic lifestyle risk screening.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json", help="survey as a JSON string")
    source.add_argument("--text", help="survey as raw/OCR'd text")
    source.add_argument("--image", help="path to a scanned survey image")
    source.add_argument("--make-sample", metavar="PATH",
                        help="render a scan-like sample form to PATH and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="include per-step output and the score breakdown")
    args = parser.parse_args(argv)

    if args.make_sample:
        print(make_sample_form(args.make_sample))
        return 0

    try:
        if args.image:
            with open(args.image, "rb") as handle:
                result = run_pipeline(image_bytes=handle.read(), verbose=args.verbose)
        else:
            payload = args.json or args.text or sys.stdin.read()
            if not payload.strip():
                parser.error("no input given (use --json/--text/--image or stdin)")
            result = run_pipeline(payload=payload, verbose=args.verbose)
    except GuardrailExit as exit_:
        print(json.dumps(exit_.payload, indent=2))
        return 2
    except OCRUnavailable as exc:
        print(json.dumps({"status": "ocr_unavailable", "reason": str(exc)}, indent=2))
        return 3
    except FileNotFoundError:
        print(json.dumps({"status": "error", "reason": f"no such file: {args.image}"}))
        return 4

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:            # e.g. `python profiler.py ... | head`
        sys.stderr.close()
        raise SystemExit(0)
