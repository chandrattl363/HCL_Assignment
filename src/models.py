"""Pydantic schemas for claim intake and decision output."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Decision = Literal["Approve", "Partially Approve", "Reject", "Manual Review"]


class LineItem(BaseModel):
    """A single expense line on a claim."""

    category: str = Field(description="Expense category, e.g. airfare, lodging, meals.")
    description: str = ""
    amount: float = Field(ge=0, description="Total amount for this line item (claim currency).")
    date: str = Field(description="Expense date, ISO YYYY-MM-DD.")
    vendor: Optional[str] = None
    receipt_attached: bool = False

    # Optional, category-specific context used by the limit checker.
    days: Optional[int] = Field(default=None, description="Number of days for per-day categories.")
    nights: Optional[int] = Field(default=None, description="Number of nights for lodging.")
    fare_class: Optional[str] = Field(default=None, description="economy / business / first for airfare.")
    booked_days_in_advance: Optional[int] = None


class Claim(BaseModel):
    """An incoming reimbursement claim."""

    claim_id: str
    employee_id: str
    employee_name: str
    purpose: str = ""
    travel_start: str
    travel_end: str
    submission_date: str
    currency: str = "INR"
    line_items: List[LineItem]

    @property
    def claimed_amount(self) -> float:
        return round(sum(item.amount for item in self.line_items), 2)


class Deduction(BaseModel):
    """An amount removed from the claimed total, with a reason."""

    item: str
    amount: float = Field(ge=0)
    reason: str


class ReimbursementDecision(BaseModel):
    """The structured recommendation returned by the agent."""

    claim_id: str
    decision: Decision
    claimed_amount: float = Field(ge=0)
    approved_amount: float = Field(ge=0)
    rejected_amount: float = Field(ge=0, description="Claimed minus approved.")
    deductions: List[Deduction] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    policy_references: List[str] = Field(
        default_factory=list, description="Policy sections / rules the decision rests on."
    )
    required_approval: str = Field(default="", description="Approval level per the approval matrix.")
    confidence: float = Field(ge=0, le=1, description="Agent confidence 0-1.")
    reason_codes: List[str] = Field(
        default_factory=list, description="Short machine-readable codes, e.g. OVER_CAP, MISSING_RECEIPT."
    )
    explanation: str = Field(description="Short human-readable justification.")
