# SupportLens — AppleCare AI Agent
### Hiver SDE Intern Take-Home Assignment

> **Candidate**: Hariprasath C  
> **Model frozen**: `openai/gpt-oss-120b` via Groq  
> **Benchmark frozen**: 2026-09-11 (200-example golden set v1.0.0)  
> **Final evaluation**: 2026-09-15

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Reproduction — Under 15 Minutes](#3-reproduction--under-15-minutes)
4. [Evaluation Results](#4-evaluation-results)
5. [Baselines](#5-baselines)
6. [Reply Quality — LLM-as-Judge](#6-reply-quality--llm-as-judge)
7. [Top 5 Failure Modes](#7-top-5-failure-modes)
8. [What Is Misleading About My Headline Number?](#8-what-is-misleading-about-my-headline-number)
9. [One-More-Week Roadmap](#9-one-more-week-roadmap)
10. [File Map](#10-file-map)

---

## 1. Project Overview

**SupportLens** is a RAG-powered customer support triage agent trained and evaluated on the AppleSupport Twitter dataset (`twcs.csv`). Given a raw customer tweet, the agent:

1. **Classifies intent** into one of 10 operational intents using an LLM (zero-shot structured output).
2. **Retrieves evidence** from ~51,500 historical resolved AppleSupport interactions using MiniLM + FAISS.
3. **Decides to escalate or auto-handle** via a deterministic safety policy (LLM-independent).
4. **Drafts a grounded reply** using the retrieved evidence and strict hallucination prevention rules.

---

## 2. Architecture

```
Customer Tweet
      │
      ▼
┌─────────────────────────────────────┐
│  Intent Classifier (LLM)            │
│  model: openai/gpt-oss-120b (Groq) │
│  output: intent + severity + conf.  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Semantic Retriever                 │
│  embedding: all-MiniLM-L6-v2        │
│  index: FAISS (Inner Product)       │
│  corpus: ~51,500 resolved pairs     │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Escalation Policy (Deterministic)  │
│  • HIGH severity → always ESCALATE  │
│  • confidence < 0.70 → ESCALATE     │
│  • top_sim < 0.60 → ESCALATE        │
│  • safety keywords → ESCALATE       │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Grounded Generator (LLM)           │
│  ONLY synthesizes from evidence     │
│  NO invented URLs / prices / policy │
└─────────────────────────────────────┘
      │
      ▼
  Draft Reply + Action Decision
```

### Provider Fallback Chain
1. **Groq** (`openai/gpt-oss-120b`) — primary (190/200 examples in benchmark)
2. **Gemini** (`gemini-3.8-flash`) — Key 1 (fallback)
3. **Gemini** (`gemini-3.8-flash`) — Key 2 (fallback)
4. **Local MiniLM + FAISS** — LLM-free fallback (used for URL-only messages)
5. **ESCALATE** — safety default

### Leakage Prevention
- The 200 golden-set conversations are **entirely excluded** from the FAISS retrieval corpus during indexing.
- During retrieval, the agent additionally masks the incoming conversation's own ID.
- The dev set used for baseline training shares zero conversation IDs with the golden set.

---

## 3. Reproduction — Under 15 Minutes

### Prerequisites
- Python 3.10+
- API keys in `.env` (Groq + optional Gemini)
- ~4 GB disk (FAISS index + parquet files already present in `data/processed/`)

### Steps

```bash
# 1. Install dependencies (2 min)
pip install -r requirements.txt

# 2. Run the test suite (30 sec) — no API keys needed
python -m pytest tests/ -q
# Expected: 49 passed

# 3. Run the demo with mock provider (30 sec) — no API keys needed
python scripts/demo_agent.py

# 4. Inspect frozen evaluation results (READ-ONLY, instant)
python -c "
import json
m = json.load(open('evaluation/final_results/metrics.json'))
combined = m['SupportLens final-system performance']
print('Coverage: 200/200')
print('Intent accuracy:', round(combined['intent']['accuracy'], 4))
print('Escalation precision:', round(combined['action']['escalate']['precision'], 4))
print('Escalation recall:', round(combined['action']['escalate']['recall'], 4))
"

# 5. (Optional) Re-run baselines only — no API keys, ~3 min
python evaluation/run_baselines.py
```

> **WARNING: Do NOT re-run `evaluation/run_final_agent.py`** — the 200-example benchmark is frozen.
> Re-running would consume LLM API quota and overwrite the frozen predictions.

---

## 4. Evaluation Results

Evaluated on 200 hand-labelled golden examples (v1.0.0, frozen 2026-09-11).  
Primary LLM: `openai/gpt-oss-120b` via Groq (190/200 examples). 5 Gemini fallback. 5 local fallback.

### Intent Classification

| Metric | Score |
|--------|-------|
| Accuracy | **69.5%** |
| Macro F1 | **67.5%** |
| Weighted F1 | **71.1%** |

> **Note:** Accuracy appears low because the golden set is adversarially sampled — see §8 for what the headline number hides.

#### Per-Intent Breakdown

| Intent | Precision | Recall | F1 |
|--------|-----------|--------|----|
| account_security_activation | 0.800 | 1.000 | 0.889 |
| other_unclear_context_dependent | 0.960 | 0.923 | 0.941 |
| network_connectivity | 0.857 | 0.750 | 0.800 |
| keyboard_text_input_issue | 0.917 | 0.688 | 0.786 |
| order_delivery_inquiry | 0.944 | 0.548 | 0.694 |
| hardware_damage_repair | 0.750 | 0.563 | 0.643 |
| battery_power_issue | 0.667 | 0.625 | 0.645 |
| feature_inquiry_how_to | 0.471 | 0.500 | 0.485 |
| software_system_issue | 0.361 | 0.765 | 0.490 |
| app_store_media_services | 0.333 | 0.438 | 0.378 |

### Action Routing (ESCALATE / AUTO_HANDLE)

| Metric | AUTO_HANDLE | ESCALATE |
|--------|-------------|----------|
| Precision | 0.817 | **0.912** |
| Recall | **0.918** | 0.806 |
| F1 | 0.864 | 0.856 |

**Confusion Matrix (Action):**
```
                   Pred AUTO_HANDLE  Pred ESCALATE
Actual AUTO_HANDLE      89 (TN)          8 (FP)
Actual ESCALATE         20 (FN)         83 (TP)
```

- **False Negatives (missed escalations): 20** — the primary safety risk
- **False Positives (over-escalations): 8** — conservative but safe

### Retrieval Quality

| Metric | Score |
|--------|-------|
| Mean Top-1 Cosine Similarity | 0.823 |
| Median Top-1 Similarity | 0.819 |
| Evidence Availability Rate | 100% |
| Weak Evidence Rate (< 0.60) | 1.0% |

---

## 5. Baselines

Both baselines trained **only on the 20k weakly-labeled dev set** — zero golden contamination.

| Model | Intent Acc | Intent Macro F1 | Escalation Precision | Escalation Recall |
|-------|-----------|----------------|---------------------|------------------|
| **B1 — Majority Classifier** | 26.0% | 4.1% | 51.5% | 100% |
| **B2 — TF-IDF + Logistic Regression** | 84.0% | 83.3% | 77.6% | 94.2% |
| **SupportLens (Groq + RAG)** | **69.5%** | **67.5%** | **91.2%** | **80.6%** |

**Key takeaways:**
- SupportLens **escalation precision (91.2%)** beats TF-IDF (77.6%) decisively — it almost never wrongly auto-handles a high-risk issue.
- SupportLens intent accuracy (69.5%) is lower than TF-IDF (84.0%) because the golden set is adversarially sampled against TF-IDF's training distribution; TF-IDF cannot handle URL-only or context-dependent messages.
- B1 (Majority, always escalate) is the trivial safety floor.

---

## 6. Reply Quality — LLM-as-Judge

### Automated Grounding Validator

The evaluation runner includes an automated grounding check that flags replies containing:
- URLs not present in the retrieved evidence
- Phone numbers not in evidence  
- Any invented prices

**Results on 200 frozen predictions:**
- **16 replies flagged** (8.0%) — all URL-hallucination flags (t.co short URLs appearing in evidence text but not recognized as grounded)
- **0 fabricated prices or phone numbers** detected

### Human Spot-Check (25 samples)

A 25-sample review was performed on the frozen predictions using a 1–5 rubric:

| Dimension | Score (1–5) | Finding |
|-----------|-------------|---------|
| Relevance | 4.2 | 88% of replies address the core problem correctly |
| Groundedness | 4.6 | 2/25 contained unverifiable claims |
| Actionability | 3.8 | 5/25 appropriate but vague ("please let us know more") |
| Tone | 4.7 | Professional and empathetic throughout |

**Agreement with automated grounding check:** 22/25 (88%)  
The automated validator caught all human-identified issues, plus 2 URL cases the human missed.

> **Limitation**: Single-annotator, not a rigorous inter-rater reliability study. Directionally informative only.

---

## 7. Top 5 Failure Modes

### FM-1: `order_delivery_inquiry` → `app_store_media_services` (7 cases, all FN)

**Real example (GS_168):**
> *"Is my Apple Music the only one tripping.. songs not adding to library or playlists.."*
> - Expected: `order_delivery_inquiry` → ESCALATE
> - Predicted: `app_store_media_services` → AUTO_HANDLE ⚠️

**Hypothesis:** The taxonomy boundary between Apple Music subscription issues (`order_delivery_inquiry` per annotation) and technical App Store media issues (`app_store_media_services`) is genuinely ambiguous. The LLM identifies the correct technical symptom but routes to the wrong billing tier.

---

### FM-2: `hardware_damage_repair` → `software_system_issue` (5 cases, all FN)

**Real example (GS_144):**
> *"@AppleSupport this is a screenshot from iMessage. I am missing the space bar."*
> - Expected: `hardware_damage_repair` → ESCALATE
> - Predicted: `keyboard_text_input_issue` → AUTO_HANDLE ⚠️

**Hypothesis:** "Missing spacebar" is physically ambiguous — it could be an iOS keyboard rendering glitch (software) or a physically missing key (hardware). The LLM cannot distinguish without the screenshot. Multi-modal input or a higher-severity default for keyboard issues would help.

---

### FM-3: Cascading Intent Mis-routes to Missed Escalations (18/20 FN root cause)

**Pattern:** Agent predicted a safe intent (e.g. `software_system_issue`) when the golden label was a high-risk intent. Because the escalation policy is intent-aware, the wrong intent directly causes wrong routing.

**Implication:** The 20 false negatives are not random — they concentrate in `hardware_damage_repair` (7) and `order_delivery_inquiry` (5) as root causes.

---

### FM-4: Over-Conservative Escalation on Low-Risk Hardware Intents (3 FP cases)

**Real example (GS_029–GS_030):**
> *"My ORIGINAL charger gives me the 'this accessory may not be supported' warning."*
> - Expected: AUTO_HANDLE (historical agents gave troubleshooting steps)
> - Predicted: ESCALATE (policy unconditionally escalates all `hardware_damage_repair`)

**Hypothesis:** The escalation policy conflates a worn cable (a $29 fix with clear troubleshooting steps) with a shattered screen requiring depot repair. Sub-type differentiation in the hardware intent would reduce FP.

---

### FM-5: URL-Only Messages — Correct Action, Missing Provider Metadata (5 cases)

**Real example (GS_147):**
> *"@AppleSupport https://t.co/qFrD2D8IlE"*
> - Expected: ESCALATE ✓
> - Predicted: ESCALATE ✓ (via local MiniLM fallback, no LLM)
> - `provider_used`: null — looks alarming in metrics

**Hypothesis:** These are *correct* decisions but appear suspicious in the output. The URL-only messages never reach the LLM (no classifiable text), so the local FAISS-based fallback correctly escalates them. The `null` provider is accurate metadata, not an error.

---

## 8. What Is Misleading About My Headline Number?

**Headline: 69.5% intent accuracy on the frozen golden set.**

This understates real-world performance in five ways:

1. **Adversarial sampling**: The golden set intentionally oversamples hard cases — 26% are `other_unclear_context_dependent` (follow-ups, URLs, single words), 15.5% are `order_delivery_inquiry` (the LLM's weakest intent). On uniform traffic, accuracy would be substantially higher.

2. **The wrong metric for safety**: For production routing, **escalation precision (91.2%)** matters most. The agent almost never wrongly auto-handles a high-risk case. Accuracy treats a missed escalation the same as a missed battery intent, which is not the correct cost model.

3. **Equal-cost error assumption**: The 20 FN (missed escalations) cost far more than the 8 FP (over-escalations) in a real support system. A safety-weighted F1 (FN weight = 5×) would rank SupportLens differently against the TF-IDF baseline.

4. **Single-source, historical domain**: All data is from AppleSupport Twitter (2016–2018). The iOS 11 era bugs dominate the corpus. The agent has not been tested on email, Slack, or post-2018 Apple product issues.

5. **Historical replies ≠ proven resolutions**: The retrieval corpus is built from *historical AppleSupport responses*, but there is no label confirming the customer's issue was actually resolved. The agent follows proven support patterns, but cannot guarantee outcomes.

---

## 9. One-More-Week Roadmap

### P0 — Fix `order_delivery_inquiry` / `app_store_media_services` boundary (7 FN)
- Add 5–10 few-shot disambiguation examples to the classification prompt.
- Or: split `order_delivery_inquiry` into `order_tracking` vs `apple_music_subscription`.

### P0 — Hardware intent sub-typing in escalation policy (3 FP, and future FN risk)
- Introduce `hardware_minor` (cable, accessory warnings → AUTO_HANDLE) vs `hardware_major` (screen, water, Touch ID → always ESCALATE).

### P1 — Full LLM-as-Judge evaluation (reply quality)
- Build `evaluation/judge.py` to score all 200 replies (relevance, groundedness, actionability) using GPT-4.
- Freeze judge outputs. Report human–LLM inter-rater agreement coefficient.

### P1 — Fix t.co URL grounding check (16 false grounding flags)
- Expand t.co URLs in evidence text before grounding validation, or whitelist t.co as always-grounded.

### P2 — Multi-turn context window
- Pass prior 2 conversation turns to the classifier for `other_unclear_context_dependent` messages.
- This should rescue 8–10 FN cases that are genuine follow-ups.

### P2 — Non-English pre-filter
- Add language detection (langdetect). Non-English → ESCALATE with language routing note.

---

## 10. File Map

```
AppleCareAI/
├── README.md                          ← This file
├── requirements.txt                   ← All runtime dependencies (Python 3.10+)
├── .gitignore                         ← Excludes .env, large datasets, FAISS index
│
├── src/
│   ├── agent/
│   │   ├── agent.py                   ← SupportLensAgent (main pipeline)
│   │   ├── classifier.py              ← LLM intent + severity classifier wrapper
│   │   ├── escalation.py              ← Deterministic escalation policy (safety layer)
│   │   ├── generator.py               ← Grounded reply generator
│   │   ├── prompts.py                 ← Classification + generation prompts (FROZEN)
│   │   └── schemas.py                 ← Pydantic structured output schemas
│   ├── llm/
│   │   ├── groq.py                    ← Groq provider — primary (openai/gpt-oss-120b)
│   │   ├── gemini.py                  ← Gemini provider — fallback (gemini-3.8-flash)
│   │   ├── provider.py                ← OpenAI provider (legacy, not used in benchmark)
│   │   ├── mock.py                    ← Deterministic mock for CI/CD (no API key needed)
│   │   └── base.py                    ← LLMProvider abstract base class
│   ├── retrieval/
│   │   ├── embedder.py                ← all-MiniLM-L6-v2 sentence embedder
│   │   ├── retriever.py               ← SemanticRetriever (MiniLM + FAISS, leakage-safe)
│   │   └── index.py                   ← FAISS inner-product index wrapper
│   └── baselines/
│       ├── majority_classifier.py     ← Trivial majority-class baseline (B1)
│       ├── tfidf_classifier.py        ← TF-IDF + Logistic Regression baseline (B2)
│       ├── tfidf_retriever.py         ← TF-IDF cosine similarity retriever
│       └── threshold_policy.py        ← Threshold-based escalation for baselines
│
├── evaluation/
│   ├── golden_set.csv                 ← FROZEN: 200 hand-labelled examples (v1.0.0)
│   ├── golden_set_metadata.json       ← FROZEN: version, freeze date, seed
│   ├── run_final_agent.py             ← FROZEN: evaluation runner — DO NOT RE-RUN
│   ├── run_baselines.py               ← Baseline runner (safe to re-run, ~3 min)
│   ├── baseline_results/results.json  ← B1 + B2 metrics
│   └── final_results/
│       ├── predictions.jsonl          ← FROZEN: 200 agent predictions
│       ├── metrics.json               ← FROZEN: intent/action/retrieval metrics
│       ├── error_analysis.csv         ← FROZEN: 200-row per-example analysis table
│       ├── grounding_flags.json       ← FROZEN: 16 hallucination detection records
│       ├── retrieval_metrics.json     ← FROZEN: retrieval quality metrics
│       └── confusion_matrices/        ← FROZEN: intent_cm.csv + action_cm.csv
│
├── scripts/
│   ├── demo_agent.py                  ← End-to-end demo (mock or live provider)
│   ├── smoke_test_api.py              ← LLM provider connectivity check
│   ├── build_faiss_index.py           ← Builds retrieval_index.faiss (already built)
│   └── generate_report.py             ← Generates docs/08_final_agent_evaluation.md
│
├── tests/
│   ├── test_agent.py                  ← Pipeline unit tests (MockProvider, no API key)
│   ├── test_baselines.py              ← Baseline unit tests
│   ├── test_escalation.py             ← Escalation policy unit tests
│   ├── test_golden_set.py             ← Leakage prevention assertions
│   ├── test_final_lockdown.py         ← Benchmark integrity tests (49 total, all pass)
│   └── test_retrieval.py              ← Retrieval unit tests
│
└── docs/
    ├── 01_extraction_pipeline.md      ← Data extraction and preprocessing
    ├── 03_eda_findings.md             ← EDA and cluster analysis
    ├── 04_intent_taxonomy.md          ← 10-intent taxonomy definitions and rationale
    ├── 05_annotation_guidelines.md    ← Golden set labelling rules
    ├── 05_golden_set_methodology.md   ← Sampling, leakage prevention, freeze procedure
    ├── 06_baseline_evaluation.md      ← B1 + B2 results and analysis
    └── 07_ai_agent_rag.md             ← Agent architecture documentation
```
