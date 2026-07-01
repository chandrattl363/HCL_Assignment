"""Offline tests for the deterministic rules engine (no LLM / API key needed).

These pin the business-correct outcome for each sample claim, independent of the LLM.
Run with:  pytest -q
"""

import json
from pathlib import Path

import pytest

from src.models import Claim
from src.policy_store import PolicyStore
from src.validator import deterministic_review

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_claims.json"

EXPECTED = {
    "CLM-001": {"decision": "Approve", "approved": 35500.0},
    "CLM-002": {"decision": "Partially Approve", "approved": 122500.0},
    "CLM-003": {"decision": "Reject", "approved": 0.0},
    "CLM-004": {"decision": "Manual Review", "approved": None},   # amount provisional
    "CLM-005": {"decision": "Reject", "approved": 0.0},
}


@pytest.fixture(scope="module")
def policy():
    return PolicyStore()


@pytest.fixture(scope="module")
def claims():
    return {c["claim_id"]: c for c in json.loads(DATA.read_text(encoding="utf-8"))}


@pytest.mark.parametrize("claim_id", list(EXPECTED))
def test_decision_label(claim_id, claims, policy):
    claim = Claim.model_validate(claims[claim_id])
    review = deterministic_review(claim, policy)
    assert review["decision"] == EXPECTED[claim_id]["decision"]


@pytest.mark.parametrize("claim_id", ["CLM-001", "CLM-002", "CLM-003", "CLM-005"])
def test_approved_amount(claim_id, claims, policy):
    claim = Claim.model_validate(claims[claim_id])
    review = deterministic_review(claim, policy)
    assert review["approved_amount"] == EXPECTED[claim_id]["approved"]


def test_duplicate_detected(claims, policy):
    claim = Claim.model_validate(claims["CLM-005"])
    review = deterministic_review(claim, policy)
    assert "DUPLICATE" in review["reason_codes"]


def test_partial_has_deductions(claims, policy):
    claim = Claim.model_validate(claims["CLM-002"])
    review = deterministic_review(claim, policy)
    assert review["deductions"]
    assert review["rejected_amount"] == pytest.approx(14500.0)


def test_manual_review_flags(claims, policy):
    claim = Claim.model_validate(claims["CLM-004"])
    review = deterministic_review(claim, policy)
    assert review["decision"] == "Manual Review"
    assert "MISSING_MATERIAL_RECEIPT" in review["reason_codes"]
    assert "BUSINESS_CLASS_UNAPPROVED" in review["reason_codes"]
