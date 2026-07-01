"""FastAPI wrapper around the Travel Reimbursement Approval Agent.

Exposes the same `evaluate_claim` pipeline used by the CLI over HTTP. Run with:

    uvicorn src.api:app --reload

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import evaluate_claim
from .models import Claim, ReimbursementDecision
from .policy_store import PolicyStore

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(
    title="Travel Reimbursement Approval Agent",
    description="Evaluate a travel expense claim against policy and return a structured decision.",
    version="1.0.0",
)

_policy: PolicyStore | None = None


def get_policy() -> PolicyStore:
    global _policy
    if _policy is None:
        _policy = PolicyStore()
    return _policy


class EvaluationResponse(BaseModel):
    decision: ReimbursementDecision
    audit_trail: List[dict]


@app.get("/health")
def health() -> dict:
    policy = get_policy()
    return {"status": "ok", "retriever": policy.retriever_kind}


@app.get("/sample-claims", response_model=List[Claim])
def list_sample_claims() -> list:
    """Return the bundled sample claims (handy for trying the API)."""
    return json.loads((DATA_DIR / "sample_claims.json").read_text(encoding="utf-8"))


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(claim: Claim) -> dict:
    """Evaluate a single claim. The request body is validated against the Claim schema,
    so malformed input returns 422 automatically."""
    return evaluate_claim(claim.model_dump(), policy=get_policy())


@app.post("/evaluate/sample/{claim_id}", response_model=EvaluationResponse)
def evaluate_sample(claim_id: str) -> dict:
    """Evaluate one of the bundled sample claims by id (e.g. CLM-002)."""
    claims = json.loads((DATA_DIR / "sample_claims.json").read_text(encoding="utf-8"))
    match = next((c for c in claims if c["claim_id"] == claim_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No sample claim with id {claim_id}")
    return evaluate_claim(match, policy=get_policy())
