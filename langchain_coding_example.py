#!/usr/bin/env python3
"""Minimal example: run the Commander / Writer / Safeguard LangGraph workflow.

Usage (from repo root):
  export OPENAI_API_KEY=...
  python3 example_commander_writer_safeguard.py

Optional:
  python3 example_commander_writer_safeguard.py --query "Your question here"
  python3 example_commander_writer_safeguard.py --model gpt-5.4 --max-iterations 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from coding_scenario.langchain_mas import CommanderWriterSafeguardSystem

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CommanderWriterSafeguardSystem once.")
    parser.add_argument(
        "--query",
        default="Write Python that prints the sum of the first five primes.",
        help="User question passed to the Commander.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_GPT_MODEL", "gpt-5.4"),
        help="OpenAI chat model id (default: OPENAI_GPT_MODEL or gpt-5.4).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max retries for steps 3–6 in coding_workflow.md.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full workflow state JSON after the run.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to your environment or .env file.", file=sys.stderr)
        return 1

    system = CommanderWriterSafeguardSystem(
        model_id=args.model,
        max_iterations=args.max_iterations,
    )
    result = system.answer(args.query)

    print("\n=== Final answer (Commander, step 8) ===\n")
    print(result.get("final_answer", ""))

    if args.verbose:
        print("\n=== Full result (verbose) ===\n")
        # Omit non-JSON-serializable values if any
        safe = {k: v for k, v in result.items() if k != "finished"}
        print(json.dumps(safe, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
