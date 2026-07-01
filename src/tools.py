"""Deterministic check functions and their LangChain tool wrappers."""

from __future__ import annotations

from datetime import date
from typing import Callable, Dict, List

from langchain_core.tools import StructuredTool, tool

from .models import Claim
from .policy_store import PolicyStore


def _parse(d: str) -> date:
    return date.fromisoformat(d)


def _cap_for(item, policy: PolicyStore) -> float | None:
    """Return the absolute cap for a line item given its category and day/night count."""
    limits = policy.category_limits
    cat = item.category
    if cat not in limits:
        return None
    rule = limits[cat]
    basis, limit = rule["basis"], rule["limit"]
    if basis == "per_day":
        return limit * (item.days or 1)
    if basis == "per_night":
        return limit * (item.nights or 1)
    return limit  # per_trip


# --------------------------------------------------------------------------- checks


def compute_receipt_completeness(claim: Claim, policy: PolicyStore) -> Dict:
    threshold = policy.receipt_required_above
    missing = [
        {"item": it.description or it.category, "amount": it.amount, "category": it.category}
        for it in claim.line_items
        if it.amount > threshold and not it.receipt_attached
    ]
    return {
        "receipt_required_above": threshold,
        "missing_receipts": missing,
        "all_receipts_present": len(missing) == 0,
    }


def compute_eligibility(claim: Claim, policy: PolicyStore) -> Dict:
    ineligible = [
        {"item": it.description or it.category, "category": it.category, "amount": it.amount}
        for it in claim.line_items
        if it.category in policy.non_reimbursable_categories
        or it.category not in policy.eligible_categories
    ]
    return {
        "eligible_categories": policy.eligible_categories,
        "ineligible_items": ineligible,
        "all_items_eligible": len(ineligible) == 0,
    }


def compute_spending_limits(claim: Claim, policy: PolicyStore) -> Dict:
    overages = []
    for it in claim.line_items:
        if it.category in policy.non_reimbursable_categories:
            continue
        cap = _cap_for(it, policy)
        if cap is not None and it.amount > cap:
            overages.append(
                {
                    "item": it.description or it.category,
                    "category": it.category,
                    "claimed": it.amount,
                    "cap": round(cap, 2),
                    "overage": round(it.amount - cap, 2),
                }
            )
    return {
        "over_cap_items": overages,
        "total_overage": round(sum(o["overage"] for o in overages), 2),
        "all_within_limits": len(overages) == 0,
    }


def compute_submission_window(claim: Claim, policy: PolicyStore) -> Dict:
    days_late = (_parse(claim.submission_date) - _parse(claim.travel_end)).days
    window = policy.submission_window_days
    return {
        "days_since_travel_end": days_late,
        "submission_window_days": window,
        "within_window": days_late <= window,
    }


def compute_airfare_class(claim: Claim, policy: PolicyStore) -> Dict:
    flagged = [
        {"item": it.description or "airfare", "fare_class": it.fare_class, "amount": it.amount}
        for it in claim.line_items
        if it.category == "airfare" and (it.fare_class or "economy").lower() != "economy"
    ]
    return {
        "premium_cabin_items": flagged,
        "requires_manager_approval": len(flagged) > 0,
    }


def compute_duplicate(claim: Claim, policy: PolicyStore) -> Dict:
    paid = policy.history.get("paid_line_items", [])
    matches = []
    for it in claim.line_items:
        for prev in paid:
            if (
                prev["employee_id"] == claim.employee_id
                and (it.vendor or "") == prev["vendor"]
                and it.date == prev["date"]
                and abs(it.amount - prev["amount"]) < 0.01
            ):
                matches.append(
                    {
                        "item": it.description or it.category,
                        "amount": it.amount,
                        "matched_paid_claim": prev["claim_id"],
                    }
                )
    return {"suspected_duplicates": matches, "is_duplicate": len(matches) > 0}


def compute_approval_threshold(claim: Claim, policy: PolicyStore, net_amount: float | None = None) -> Dict:
    amount = claim.claimed_amount if net_amount is None else net_amount
    return {
        "net_amount": round(amount, 2),
        "required_approval": policy.required_approval(amount),
        "director_threshold": policy.director_threshold,
        "exceeds_director_threshold": amount > policy.director_threshold,
    }


def compute_settlement(claim: Claim, policy: PolicyStore) -> Dict:
    """Canonical, deterministic settlement used by the output validator.

    Approved = sum of eligible line items, each capped at its category limit, with
    non-reimbursable items removed entirely. This is the ground truth the LLM's number is
    checked against.
    """
    deductions: List[Dict] = []
    approved = 0.0
    for it in claim.line_items:
        if it.category in policy.non_reimbursable_categories or it.category not in policy.eligible_categories:
            deductions.append(
                {"item": it.description or it.category, "amount": it.amount, "reason": "ineligible category"}
            )
            continue
        cap = _cap_for(it, policy)
        allowed = it.amount if cap is None else min(it.amount, cap)
        if cap is not None and it.amount > cap:
            deductions.append(
                {
                    "item": it.description or it.category,
                    "amount": round(it.amount - cap, 2),
                    "reason": f"over cap ({cap})",
                }
            )
        approved += allowed
    approved = round(approved, 2)
    return {
        "claimed_amount": claim.claimed_amount,
        "approved_amount": approved,
        "rejected_amount": round(claim.claimed_amount - approved, 2),
        "deductions": deductions,
    }


# --------------------------------------------------------------------------- tool factory


def build_tools(claim: Claim, policy: PolicyStore) -> List[StructuredTool]:
    """Build LangChain tools bound to the current claim and policy.

    The check tools take no arguments — they operate on the bound claim — so the LLM's job is
    simply to decide which checks are relevant, then combine the structured results.
    """

    @tool
    def policy_lookup(query: str) -> str:
        """Look up relevant travel-policy rules for a topic such as 'meal per diem',
        'airfare class', 'receipt requirement', 'submission deadline', or 'approval matrix'."""
        return policy.search(query)

    @tool
    def check_receipt_completeness() -> Dict:
        """Check the claim for line items over the receipt threshold that have no receipt
        attached. Returns the list of missing documents."""
        return compute_receipt_completeness(claim, policy)

    @tool
    def check_eligibility() -> Dict:
        """Check each line item against the eligible and non-reimbursable category lists.
        Returns any ineligible items that must be removed."""
        return compute_eligibility(claim, policy)

    @tool
    def check_spending_limits() -> Dict:
        """Compare each line item against its per-diem / per-night / per-trip cap. Returns
        over-cap items and the overage amounts that should be deducted."""
        return compute_spending_limits(claim, policy)

    @tool
    def check_submission_window() -> Dict:
        """Check whether the claim was submitted within the policy window after travel end."""
        return compute_submission_window(claim, policy)

    @tool
    def check_airfare_class() -> Dict:
        """Check for business/first-class airfare that requires prior manager approval."""
        return compute_airfare_class(claim, policy)

    @tool
    def detect_duplicate() -> Dict:
        """Check the claim's line items against previously paid claims for duplicates."""
        return compute_duplicate(claim, policy)

    @tool
    def check_approval_threshold() -> Dict:
        """Return the approval level required for the claim amount per the approval matrix."""
        return compute_approval_threshold(claim, policy)

    return [
        policy_lookup,
        check_receipt_completeness,
        check_eligibility,
        check_spending_limits,
        check_submission_window,
        check_airfare_class,
        detect_duplicate,
        check_approval_threshold,
    ]
