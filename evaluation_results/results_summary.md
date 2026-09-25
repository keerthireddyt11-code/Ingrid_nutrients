### Overall (20 cases)

| Variant | Correctness | Grounding | Safety | Clarity | Verdict accuracy | No verdict | Forbidden verdicts | Banned phrases | Errors |
|---|---|---|---|---|---|---|---|---|---|
| A - Minimal LLM | 2.26 | 2.89 | 4.79 | 4.84 | 7/19 | 2 | 0 | 0 | 1 |
| B - Simplified | 3.85 | 4.2 | 4.05 | 4.55 | 15/20 | 0 | 0 | 0 | 0 |
| C - Full system | 4.75 | 4.55 | 5.0 | 4.25 | 20/20 | 0 | 0 | 1 | 0 |

### Category: ambiguous

| Variant | Correctness | Grounding | Safety | Clarity | Verdict accuracy | No verdict | Forbidden verdicts | Banned phrases | Errors |
|---|---|---|---|---|---|---|---|---|---|
| A - Minimal LLM | 1.0 | 2.5 | 5.0 | 5.0 | 0/4 | 0 | 0 | 0 | 0 |
| B - Simplified | 1.25 | 3.0 | 2.25 | 3.5 | 1/4 | 0 | 0 | 0 | 0 |
| C - Full system | 4.25 | 3.5 | 5.0 | 4.25 | 4/4 | 0 | 0 | 0 | 0 |

### Category: misleading_instruction

| Variant | Correctness | Grounding | Safety | Clarity | Verdict accuracy | No verdict | Forbidden verdicts | Banned phrases | Errors |
|---|---|---|---|---|---|---|---|---|---|
| A - Minimal LLM | 4.2 | 4.2 | 4.2 | 4.8 | 5/5 | 2 | 0 | 0 | 1 |
| B - Simplified | 5.0 | 5.0 | 5.0 | 5.0 | 6/6 | 0 | 0 | 0 | 0 |
| C - Full system | 4.83 | 5.0 | 5.0 | 3.5 | 6/6 | 0 | 0 | 1 | 0 |

### Category: missing_information

| Variant | Correctness | Grounding | Safety | Clarity | Verdict accuracy | No verdict | Forbidden verdicts | Banned phrases | Errors |
|---|---|---|---|---|---|---|---|---|---|
| A - Minimal LLM | 3.0 | 3.25 | 5.0 | 4.75 | 2/4 | 0 | 0 | 0 | 0 |
| B - Simplified | 4.0 | 4.25 | 4.0 | 4.5 | 3/4 | 0 | 0 | 0 | 0 |
| C - Full system | 4.75 | 4.25 | 5.0 | 4.75 | 4/4 | 0 | 0 | 0 | 0 |

### Category: normal

| Variant | Correctness | Grounding | Safety | Clarity | Verdict accuracy | No verdict | Forbidden verdicts | Banned phrases | Errors |
|---|---|---|---|---|---|---|---|---|---|
| A - Minimal LLM | 1.0 | 1.83 | 5.0 | 4.83 | 0/6 | 0 | 0 | 0 | 0 |
| B - Simplified | 4.33 | 4.17 | 4.33 | 4.83 | 5/6 | 0 | 0 | 0 | 0 |
| C - Full system | 5.0 | 5.0 | 5.0 | 4.67 | 6/6 | 0 | 0 | 0 | 0 |


_Judge model: anthropic/claude-haiku-4.5. Scores are 1-5 (LLM judge, temperature 0, blind to variant). Verdict accuracy, no-verdict, forbidden-verdict, banned-phrase and error counts are computed in code._