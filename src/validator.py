"""Deterministic rules engine that computes the authoritative claim decision."""

from __future__ import annotations

from typing import Dict

from .models import Claim
from .policy_store import PolicyStore
from .tools import (
    compute_airfare_class,
    compute_duplicate,
    compute_eligibility,
    compute_receipt_completeness,
    compute_settlement,
    compute_spending_limits,
    compute_submission_window,
)

MATERIAL_RECEIPT_AMOUNT = 10000.0  # missing receipts at/above this route to manual review

# Map reason codes to the policy sections that justify them (rule basis / grounding).
POLICY_REF = {
    "DUPLICATE": "Policy Section 8: Duplicate Claims",
    "OUTSIDE_WINDOW": "Policy Section 6: Submission Window",
    "OVER_CAP": "Policy Section 3: Per-Diem and Category Limits",
    "INELIGIBLE_ITEM": "Policy Section 2: Non-Reimbursable Categories",
    "MISSING_RECEIPT": "Policy Section 5: Receipt Requirements",
    "MISSING_MATERIAL_RECEIPT": "Policy Section 5: Receipt Requirements",
    "BUSINESS_CLASS_UNAPPROVED": "Policy Section 4: Airfare Rules",
    "EXCEEDS_DIRECTOR_THRESHOLD": "Policy Section 7: Approval Matrix",
    "AUTO_APPROVE": "Policy Section 7: Approval Matrix",
}


def deterministic_review(claim: Claim, policy: PolicyStore) -> Dict:
    """Compute the authoritative decision for a claim from the rules engine."""
    settlement = compute_settlement(claim, policy)
    eligibility = compute_eligibility(claim, policy)
    limits = compute_spending_limits(claim, policy)
    receipts = compute_receipt_completeness(claim, policy)
    window = compute_submission_window(claim, policy)
    airfare = compute_airfare_class(claim, policy)
    duplicate = compute_duplicate(claim, policy)

    deductions = list(settlement["deductions"])
    reason_codes: list[str] = []
    missing_documents: list[str] = []

    # --- small (immaterial) missing receipts are deducted; material ones flag manual review ---
    material_missing = False
    for item in receipts["missing_receipts"]:
        missing_documents.append(f"Receipt for '{item['item']}' (INR {item['amount']:.2f})")
        if item["amount"] >= MATERIAL_RECEIPT_AMOUNT:
            material_missing = True
        else:
            deductions.append(
                {"item": item["item"], "amount": item["amount"], "reason": "missing receipt"}
            )

    # Recompute approved after immaterial-receipt deductions.
    approved = round(
        settlement["claimed_amount"] - sum(d["amount"] for d in deductions), 2
    )
    approved = max(approved, 0.0)
    rejected = round(claim.claimed_amount - approved, 2)

    # --- reason codes from evidence ---
    if limits["over_cap_items"]:
        reason_codes.append("OVER_CAP")
    if not eligibility["all_items_eligible"]:
        reason_codes.append("INELIGIBLE_ITEM")
    if receipts["missing_receipts"] and not material_missing:
        reason_codes.append("MISSING_RECEIPT")

    approval_basis = policy.required_approval(approved)

    # --- decision label (priority order) ---
    if duplicate["is_duplicate"]:
        label = "Reject"
        approved, rejected = 0.0, claim.claimed_amount
        reason_codes = ["DUPLICATE"]
    elif not window["within_window"]:
        label = "Reject"
        approved, rejected = 0.0, claim.claimed_amount
        reason_codes = ["OUTSIDE_WINDOW"]
    elif approved == 0 and claim.claimed_amount > 0:
        label = "Reject"
        if "INELIGIBLE_ITEM" not in reason_codes:
            reason_codes.append("INELIGIBLE_ITEM")
    else:
        manual_codes = []
        if airfare["requires_manager_approval"]:
            manual_codes.append("BUSINESS_CLASS_UNAPPROVED")
        if approved > policy.director_threshold:
            manual_codes.append("EXCEEDS_DIRECTOR_THRESHOLD")
        if material_missing:
            manual_codes.append("MISSING_MATERIAL_RECEIPT")
        if manual_codes:
            label = "Manual Review"
            reason_codes.extend(manual_codes)
        elif deductions:
            label = "Partially Approve"
        else:
            label = "Approve"
            reason_codes.append("AUTO_APPROVE")

    policy_references = sorted({POLICY_REF[c] for c in reason_codes if c in POLICY_REF})

    return {
        "decision": label,
        "claimed_amount": claim.claimed_amount,
        "approved_amount": approved,
        "rejected_amount": rejected,
        "deductions": deductions,
        "missing_documents": missing_documents,
        "policy_references": policy_references,
        "required_approval": approval_basis,
        "reason_codes": reason_codes,
    }
