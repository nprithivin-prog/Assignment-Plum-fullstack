# AI-Powered Health Risk Profiler

Takes a lifestyle survey — typed, pasted as text, or photographed — and returns a
structured risk profile: factors, a score, and actionable guidance.

The service is deliberately **rule-based, not generative**. A health screening
score has to be explainable: every point traces back to a named factor, and every
recommendation traces back to a factor that was actually present in the answers.
Nothing here is diagnostic.

Everything lives in **`profiler.py`** — config, OCR, parsing, scoring, HTTP API,
CLI and the demo page. `test_profiler.py` holds the 33 tests.

## Quick start

```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr      # needed only for the image path

uvicorn profiler:app --reload           # http://localhost:8000/demo
```

Open `/demo` for the browser client, or `/docs` for the OpenAPI console.

Without a server:

```bash
python profiler.py --json '{"age":42,"smoker":true,"exercise":"rarely","diet":"high sugar"}'
python profiler.py --make-sample form.png     # renders a scan-like test form
python profiler.py --image form.png --verbose
cat survey.txt | python profiler.py
```

The CLI exits `0` on success, `2` when a guardrail stops the profile, `3` when an
image is submitted with no OCR engine installed.

Tests:

```bash
python -m pytest test_profiler.py -q        # 33 tests
```

Docker:

```bash
docker build -t health-profiler . && docker run -p 8000:8000 health-profiler
```

Both third-party layers are optional. Without FastAPI the module still imports
and the parsing, scoring and CLI paths all work — only the HTTP routes go away.
Without Tesseract everything works except the image path, which reports
`ocr_unavailable` rather than returning an empty profile.

## The four steps

### Step 1 — OCR / text parsing

`POST /v1/parse` (JSON or `{"text": "..."}`) · `POST /v1/parse/image` (multipart)

```json
{
  "answers": {"age": 42, "smoker": true, "exercise": "rarely", "diet": "high sugar"},
  "missing_fields": [],
  "confidence": 0.92
}
```

Both input channels converge on the same canonical answers object. Typed JSON,
pasted form text, and an actual photographed form of the same survey all produce
identical `answers`; only the confidence differs.

### Step 2 — Factor extraction

`POST /v1/factors`

```json
{"factors": ["smoking", "poor diet", "low exercise"], "confidence": 0.88}
```

### Step 3 — Risk classification

`POST /v1/risk`

```json
{"risk_level": "high", "score": 78, "rationale": ["smoking", "high sugar diet", "low activity"]}
```

### Step 4 — Recommendations

`POST /v1/recommendations`

```json
{
  "risk_level": "high",
  "factors": ["smoking", "poor diet", "low exercise"],
  "recommendations": ["Quit smoking", "Reduce sugar", "Walk 30 mins daily"],
  "status": "ok"
}
```

`POST /v1/profile` runs all four at once. Add `?verbose=true` to get each
intermediate step plus the score breakdown — that is what the demo page renders.

## Handling noisy input

Scanned forms and typed forms fail in different ways, so parsing is layered.

**Labels** are matched in three tiers: exact alias, then punctuation- and
space-insensitive alias, then edit-distance similarity above 0.72. This is what
lets `Aqe:`, `Srnoker:`, `Exercse:` and `Dlet:` still land on the right fields.
`Patient Age`, `How often do you exercise`, and `Eating habits` resolve through
the alias table.

**Values** are normalised against vocabularies rather than accepted verbatim, so
`yes` / `Y` / `true` / `1` / `daily` all become `true`, and free-text exercise
answers collapse onto an ordinal scale (`never` → `rarely` → `sometimes` →
`often` → `daily`). A numeric answer like `3 times a week` is mapped onto that
same scale.

Two traps this had to be built around, both caught by tests:

- **Negation must not be swallowed by substring matching.** `non-smoker` contains
  `smoker`, and the single-letter token `n` appears inside `occasionally`. Value
  matching is therefore whole-word, and negatives are checked before positives.
- **Digit repair must not apply to words.** OCR renders `42` as `4Z` and `50` as
  `SO`, so a repair pass exists — but applied naively it turns `abc` into `6`.
  Repair only runs on short, plausibly-numeric tokens.

**A value that cannot be mapped to anything canonical is recorded as missing, not
guessed.** In a health context an explicit gap is safer than a confident
invention. `N/A`, `-`, `not stated` and friends are gaps, not answers — notably,
`n/a` is kept out of the negative vocabulary so it never becomes a `false`.

## Confidence

Each field carries its own confidence — the label-match score multiplied by the
value-normalisation score. The reported number is their mean, scaled by two
things:

- **the input channel**: typed JSON 1.00, pasted text 0.97, image 0.93, where the
  image figure is blended with Tesseract's own per-word confidence for that scan;
- **coverage**: a form where only one field in four was recovered should not
  report 0.99 because that one field happened to be crisp.

Step 2's confidence is step 1's, weighted by the cleanliness of the specific
fields that actually drove the verdict.

## Guardrails

The pipeline stops after step 1 and returns a terminal object rather than a
half-guessed profile when either condition holds:

```json
{"status": "incomplete_profile", "reason": ">50% fields missing"}
```

```json
{"status": "incomplete_profile", "reason": "input too noisy to parse reliably"}
```

Required fields are `age`, `smoker`, `exercise`, `diet`. The threshold is a
strict *majority* missing — two of four is still answerable, three of four is
not. HTTP callers get `422`; the CLI exits `2`. If an image is submitted with no
OCR engine installed, the service says so (`503`) instead of silently returning
an empty profile.

## The scoring model

Additive, capped at 100, with every weight in the configuration section so a
clinician can tune them without touching pipeline code.

| Factor | Points |
|---|---|
| Smoking (daily) | 30 |
| Smoking (occasional) | 15 |
| No physical activity | 25 |
| Poor diet | 20 |
| Low exercise | 20 |
| Heavy alcohol use | 15 |
| Obesity (BMI ≥ 30) | 15 |
| Family history | 10 |
| Insufficient sleep (< 6h) | 8 |
| Age band | 4 – 20 |

Bands: **low** 0–25, **moderate** 26–55, **high** 56–100.

The reference profile — 42, smoker, rarely exercises, high-sugar diet — scores
30 + 20 + 20 + 8 = **78**, which is `high`.

Age is scored but reported separately from the behavioural factors, because it is
not something the person can act on and it should not appear in a list of things
to change.

Recommendations are ordered by the weight of the factor that produced them, so
the highest-impact action comes first, then deduplicated and capped at five. A
profile with no factors gets maintenance advice rather than an empty list.

## Beyond the required fields

`alcohol`, `sleep_hours`, `bmi` and `family_history` are parsed and scored when
present, and ignored when absent — they never trigger the missing-field
guardrail. Adding another optional field means adding an alias, a normaliser, a
weight and a recommendation, all in the configuration section.

## Layout

```
profiler.py        the whole service, in nine labelled sections:
  1  configuration   field schema, vocabularies, weights, advice  ← tune here
  2  OCR             image → text (multi-pass Tesseract, best read wins)
  3  parsing         text/JSON → answers + missing_fields + confidence
  4  analysis        factors, score, recommendations
  5  orchestration   pipeline + guardrails
  6  HTTP API        FastAPI routes
  7  demo page       embedded browser client
  8  sample form     renders a scan-like image for testing
  9  CLI
test_profiler.py   33 tests, including a full image round-trip
```

Section 1 is the only part a clinician or product owner should need to touch.
Adding an optional field means adding an alias, a normaliser, a weight and a
recommendation — all within that section.

OCR runs three page-segmentation modes and keeps the best read, ranked by mean
word confidence weighted by coverage — confidence alone would favour a pass that
found two crisp words over one that found twenty good ones. Preprocessing is
grayscale, median de-speckle and contrast normalisation, and it upscales only
images that are genuinely too small; upscaling a legible scan amplifies noise and
measurably made recognition worse.

## Limitations

- Scoring weights are a reasonable screening heuristic, not a validated clinical
  instrument. They would need calibration against real outcome data before being
  used for anything beyond triage.
- Factors are treated as independent and additive. Real risk factors interact.
- OCR is tuned for printed forms. Handwriting needs a different engine.
- English-language surveys only.
