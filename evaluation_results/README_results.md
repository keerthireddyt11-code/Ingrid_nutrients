# Stage 6 — Evaluation Results

Output of running the 20 test cases (see `test_cases.json` in the code folder) through variants A, B and C, then scoring them with the LLM judge.

## Files

| File | What it is |
|---|---|
| `evaluation_results.json` | Raw output of each variant (A, B, C) on all 20 test cases, before scoring. |
| `scored_results.json` | The same results, with each variant's judge scores (correctness, grounding, safety, clarity) and notes added per case. |
| `results_summary.md` | Average scores per variant, overall and by test-case category. **Start here.** |
| `results_summary.json` | Same summary as above, in machine-readable form. |

## Reading the summary

- Scores are 1–5, from an LLM judge (Claude Haiku 4.5, temperature 0, blind to which variant it's scoring).
- **Verdict accuracy** is computed in code, not by the judge — it checks whether each variant's verdict matched the expected one for that test case.
- C's 20/20 verdict accuracy is expected: the expected verdicts were derived from the app's own rubric, so C matches by construction. The informative comparison is A and B against that same policy.

Regenerating these files (via `run_evaluation.py` + `llm_judge.py` in the code folder) overwrites all four — keep a copy if you need to preserve this run as a "before" snapshot.
