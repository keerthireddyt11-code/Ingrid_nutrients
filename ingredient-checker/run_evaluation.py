"""Stage 6, bullet 2 — run every test case through variants A, B, and C, and
save all raw outputs to one file. This replaces doing it by hand."""

import json
from pathlib import Path

from variants import variant_a_minimal_llm, variant_b_simplified, variant_c_full_system

TEST_CASES_FILE = Path("test_cases.json")
OUTPUT_FILE = Path("evaluation_results.json")


def main() -> None:
    test_cases = json.loads(TEST_CASES_FILE.read_text(encoding="utf-8"))
    results = []

    for case in test_cases:
        print(f"Running case {case['id']} ({case['category']})...", flush=True)
        entry = {
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "note": case.get("note", ""),
            "A": variant_a_minimal_llm(case["input"]),
            "B": variant_b_simplified(case["input"]),
            "C": variant_c_full_system(case["input"]),
        }
        results.append(entry)

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results)} cases x 3 variants to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
