"""Command-line runner for the Travel Reimbursement Approval Agent.

    python -m src.cli --all
    python -m src.cli --id CLM-002
    python -m src.cli --file path/to/claim.json
    python -m src.cli --all --save outputs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Windows consoles default to cp1252, which can't encode the table/box characters.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from tabulate import tabulate

from .agent import evaluate_claim
from .policy_store import PolicyStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_claims(args) -> List[dict]:
    if args.file:
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    claims = json.loads((DATA_DIR / "sample_claims.json").read_text(encoding="utf-8"))
    if args.all:
        return claims
    if args.id:
        match = [c for c in claims if c["claim_id"] == args.id]
        if not match:
            sys.exit(f"No sample claim with id {args.id}")
        return match
    return claims[:1]


def _print_decision(result: dict) -> None:
    d = result["decision"]
    print("\n" + "=" * 70)
    print(f"  CLAIM {d['claim_id']}  ->  {d['decision'].upper()}  "
          f"(confidence {d['confidence']})")
    print("=" * 70)

    summary = [
        ["Claimed", f"{d['claimed_amount']:.2f}"],
        ["Approved", f"{d['approved_amount']:.2f}"],
        ["Rejected/Deducted", f"{d['rejected_amount']:.2f}"],
        ["Required approval", d["required_approval"]],
    ]
    print(tabulate(summary, tablefmt="github"))

    if d["deductions"]:
        print("\nDeductions:")
        print(tabulate(
            [[x["item"], f"{x['amount']:.2f}", x["reason"]] for x in d["deductions"]],
            headers=["Item", "Amount", "Reason"], tablefmt="github",
        ))
    if d["missing_documents"]:
        print("\nMissing documents:")
        for m in d["missing_documents"]:
            print(f"  - {m}")
    if d["policy_references"]:
        print("\nPolicy basis: " + "; ".join(d["policy_references"]))
    if d["reason_codes"]:
        print("Reason codes: " + ", ".join(d["reason_codes"]))
    print(f"\nExplanation: {d['explanation']}")

    if result.get("audit_trail"):
        print("\nAudit trail (tools called):")
        for step in result["audit_trail"]:
            args = step.get("args") or {}
            arg_str = json.dumps(args) if args else ""
            print(f"  - {step['tool']}{(' ' + arg_str) if arg_str else ''}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Travel Reimbursement Approval Agent")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Run all bundled sample claims.")
    g.add_argument("--id", help="Run a single bundled sample claim by id (e.g. CLM-002).")
    g.add_argument("--file", help="Run claim(s) from a JSON file (object or list).")
    parser.add_argument("--save", help="Directory to write JSON results into.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of tables.")
    args = parser.parse_args()

    policy = PolicyStore()
    claims = _load_claims(args)
    results = []
    for claim in claims:
        result = evaluate_claim(claim, policy=policy)
        results.append(result)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_decision(result)

    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            cid = result["decision"]["claim_id"]
            (out_dir / f"{cid}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        print(f"\nSaved {len(results)} result(s) to {out_dir}/")


if __name__ == "__main__":
    main()
