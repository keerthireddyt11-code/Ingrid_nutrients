Evaluation

We tested the ingredient checker against 20 barcode-based test cases (normal, ambiguous, missing-information and misleading/prompt-injection cases).

Each case was run through three variants:
- **A — Minimal LLM:** no database, no rubric. The model answers from its own knowledge.
- **B — Simplified system:** real product retrieval, but the LLM picks the verdict itself.
- **C — Full system:** our actual app — retrieval + rubric decides the verdict, LLM only explains it.

Each variant's output was then scored by an **LLM judge** on four criteria: **correctness, grounding, safety, clarity** (1–5 each). We also track verdict accuracy, forbidden verdicts, banned phrases and errors in code.

## Files

| File | What it is |
|---|---|
| `chain.py` | The main app pipeline (retrieval → rubric → LLM explanation). |
| `test_cases.json` | The 20 test cases, each with an expected verdict. |
| `variants.py` | Runs a given input through variant A, B and C. |
| `run_evaluation.py` | Runs all 20 test cases through A, B and C → `evaluation_results.json`. |
| `llm_judge.py` | Scores every result with the LLM judge → `scored_results.json`, `results_summary.md`/`.json`. |
| `eval_utils.py` | Shared helpers (parsing verdicts, checking against expected verdicts). |

## How to run

```
python run_evaluation.py
python llm_judge.py
```

Requires `OPENROUTER_API_KEY` (and the app's usual keys) in `.env`. `.env` is not committed.
