"""End-to-end evaluation of the live agent over the sample claims.

Requires GROQ_API_KEY (calls the Groq-hosted LLM). Skipped automatically when no key is set, so
the offline suite still runs in CI. Run explicitly with:

    pytest tests/test_agent_eval.py -q -s
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live LLM eval.",
)

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_claims.json"

EXPECTED_DECISION = {
    "CLM-001": "Approve",
    "CLM-002": "Partially Approve",
    "CLM-003": "Reject",
    "CLM-004": "Manual Review",
    "CLM-005": "Reject",
}


@pytest.fixture(scope="module")
def claims():
    return {c["claim_id"]: c for c in json.loads(DATA.read_text(encoding="utf-8"))}


@pytest.mark.parametrize("claim_id,expected", EXPECTED_DECISION.items())
def test_agent_decision(claim_id, expected, claims):
    from src.agent import evaluate_claim
    from src.policy_store import PolicyStore

    result = evaluate_claim(claims[claim_id], policy=PolicyStore())
    decision = result["decision"]
    # The deterministic validator makes the final label authoritative, so this is stable.
    assert decision["decision"] == expected
    # The agent must actually have used tools (agentic behaviour).
    assert result["audit_trail"], "agent called no tools"
