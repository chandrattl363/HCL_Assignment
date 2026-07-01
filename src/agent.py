"""LangGraph workflow: intake -> agent (LLM + tools) -> finalize"""

from __future__ import annotations

import os
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .models import Claim, Deduction, ReimbursementDecision
from .policy_store import PolicyStore
from .tools import build_tools
from .validator import deterministic_review

load_dotenv()  # ensure .env (LLM_MODEL, GROQ_API_KEY) is loaded before we read it

DEFAULT_MODEL_SPEC = "groq:llama-3.3-70b-versatile"


def _resolve_model_spec(model: Optional[str] = None) -> str:
    """Resolve a provider-qualified model spec, e.g. 'groq:llama-3.3-70b-versatile'.

    Priority: explicit arg > LLM_MODEL env > DEFAULT_MODEL_SPEC. A bare model name is assumed
    to be a Groq model.
    """
    spec = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL_SPEC
    return spec if ":" in spec else f"groq:{spec}"

SYSTEM_PROMPT = """You are a Travel Reimbursement Approval Agent for Acme Corp.

Your job: evaluate one expense claim against company policy and recommend exactly one of:
Approve, Partially Approve, Reject, or Manual Review.

You have tools that read the current claim and the policy. Call all relevant checks together
in a single batch (parallel tool calls) before deciding. At minimum run:
  - check_eligibility, check_spending_limits, check_receipt_completeness,
    check_submission_window, detect_duplicate, check_airfare_class, check_approval_threshold
  - use policy_lookup when you need the exact wording of a rule.

Decision guidance (the validator enforces the final numbers, but reason carefully):
  - Approve: fully compliant, within caps, receipts present, net amount auto-approvable.
  - Partially Approve: compliant overall but with deductions (over-cap or ineligible items).
  - Reject: duplicate claim, submitted outside the window, or nothing reimbursable remains.
  - Manual Review: business/first-class airfare without approval, a missing receipt on a
    material (>= INR 10,000) item, net amount above the director threshold, or genuinely
    ambiguous or conflicting evidence. When in doubt, prefer Manual Review over forcing a decision.

Call tools first. Once you have enough evidence, stop calling tools and say you are ready to
decide."""

FINALIZE_PROMPT = """Based on the tool evidence above, produce the structured reimbursement
decision for claim {claim_id}. Use the policy rules and the evidence; do not invent numbers.
Set a confidence between 0 and 1 (lower it when evidence is missing or conflicting), include
short reason codes, the policy references the decision rests on, and a 1-3 sentence
explanation."""


class AgentState(TypedDict):
    raw_claim: dict
    claim: Optional[Claim]
    messages: Annotated[list, add_messages]
    audit: List[dict]
    decision: Optional[ReimbursementDecision]
    intake_error: Optional[str]
    llm_error: Optional[str]


def _template_explanation(truth: dict, claim: Claim) -> str:
    """Deterministic explanation used as the baseline / LLM-unavailable fallback."""
    parts = [
        f"{truth['decision']}: approved {claim.currency} {truth['approved_amount']:.2f} "
        f"of {truth['claimed_amount']:.2f} claimed."
    ]
    if truth["deductions"]:
        parts.append(
            "Deductions — "
            + "; ".join(f"{d['item']} INR {d['amount']:.2f} ({d['reason']})" for d in truth["deductions"])
            + "."
        )
    if truth["missing_documents"]:
        parts.append("Missing: " + "; ".join(truth["missing_documents"]) + ".")
    if truth["policy_references"]:
        parts.append("Basis: " + "; ".join(truth["policy_references"]) + ".")
    return " ".join(parts)


def _get_llm(model: Optional[str] = None):
    # max_retries gives backoff on transient rate limits.
    return init_chat_model(_resolve_model_spec(model), temperature=0, max_retries=6)


def build_graph(policy: PolicyStore, model: Optional[str] = None):
    """Compile the LangGraph workflow. Tools are rebuilt per claim inside the nodes."""
    llm = _get_llm(model)

    # ---------------- nodes ----------------

    def intake(state: AgentState) -> dict:
        try:
            claim = Claim.model_validate(state["raw_claim"])
        except Exception as exc:  # malformed claim -> safe fallback
            return {"claim": None, "intake_error": str(exc)}
        msg = HumanMessage(
            content=(
                f"Evaluate this claim. Use your tools, then decide.\n\n"
                f"Claim {claim.claim_id} — {claim.employee_name} ({claim.employee_id})\n"
                f"Purpose: {claim.purpose}\n"
                f"Travel: {claim.travel_start} to {claim.travel_end}; "
                f"Submitted: {claim.submission_date}\n"
                f"Claimed total: {claim.currency} {claim.claimed_amount}\n"
                f"Line items:\n"
                + "\n".join(
                    f"  - {it.category}: {it.description} | {it.amount} {claim.currency} | "
                    f"date {it.date} | vendor {it.vendor} | "
                    f"receipt={'yes' if it.receipt_attached else 'NO'}"
                    + (f" | fare_class={it.fare_class}" if it.fare_class else "")
                    + (f" | nights={it.nights}" if it.nights else "")
                    + (f" | days={it.days}" if it.days else "")
                    for it in claim.line_items
                )
            )
        )
        return {
            "claim": claim,
            "messages": [SystemMessage(content=SYSTEM_PROMPT), msg],
            "audit": [],
        }

    def agent(state: AgentState) -> dict:
        tools = build_tools(state["claim"], policy)
        try:
            response = llm.bind_tools(tools).invoke(state["messages"])
            return {"messages": [response]}
        except Exception as exc:  # LLM/quota failure -> proceed; finalize falls back to rules
            return {
                "messages": [AIMessage(content=f"[LLM unavailable: {exc}]")],
                "llm_error": str(exc),
            }

    def record_audit(state: AgentState) -> dict:
        """Capture which tools were called (audit trail) from the latest AI message."""
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        entries = [{"tool": c["name"], "args": c.get("args", {})} for c in calls]
        return {"audit": state.get("audit", []) + entries}

    def finalize(state: AgentState) -> dict:
        claim = state["claim"]
        # The deterministic rules engine is always the source of truth for amounts/label.
        truth = deterministic_review(claim, policy)

        extra_codes: list[str] = []
        proposed_codes: list[str] = []
        proposed_refs: list[str] = []
        explanation = _template_explanation(truth, claim)
        confidence = 0.5  # deterministic-only baseline

        if not state.get("llm_error"):
            try:
                structured_llm = llm.with_structured_output(ReimbursementDecision)
                prompt = state["messages"] + [
                    HumanMessage(content=FINALIZE_PROMPT.format(claim_id=claim.claim_id))
                ]
                proposed: ReimbursementDecision = structured_llm.invoke(prompt)
                diverged = proposed.decision != truth["decision"]
                explanation = proposed.explanation
                confidence = round(proposed.confidence * (0.6 if diverged else 1.0), 2)
                proposed_codes = proposed.reason_codes
                proposed_refs = proposed.policy_references
                if diverged:
                    extra_codes.append("VALIDATOR_OVERRIDE")
            except Exception as exc:  # quota/LLM failure at finalize -> deterministic fallback
                extra_codes.append("LLM_UNAVAILABLE")
                explanation += f" (LLM unavailable at finalize: {exc}.)"
        else:
            extra_codes.append("LLM_UNAVAILABLE")

        reason_codes = list(dict.fromkeys(truth["reason_codes"] + proposed_codes + extra_codes))
        final = ReimbursementDecision(
            claim_id=claim.claim_id,
            decision=truth["decision"],  # authoritative label
            claimed_amount=truth["claimed_amount"],
            approved_amount=truth["approved_amount"],
            rejected_amount=truth["rejected_amount"],
            deductions=[Deduction(**d) for d in truth["deductions"]],
            missing_documents=truth["missing_documents"],
            policy_references=sorted(set(truth["policy_references"] + proposed_refs)),
            required_approval=truth["required_approval"],
            confidence=confidence,
            reason_codes=reason_codes,
            explanation=explanation,
        )
        return {"decision": final}

    def manual_fallback(state: AgentState) -> dict:
        """Used only when intake fails — never force a decision on invalid input."""
        cid = state["raw_claim"].get("claim_id", "UNKNOWN")
        return {
            "decision": ReimbursementDecision(
                claim_id=cid,
                decision="Manual Review",
                claimed_amount=0.0,
                approved_amount=0.0,
                rejected_amount=0.0,
                confidence=0.0,
                reason_codes=["INTAKE_VALIDATION_FAILED"],
                explanation=f"Claim could not be parsed/validated: {state.get('intake_error')}",
            )
        }

    # ---------------- edges ----------------

    def route_after_intake(state: AgentState) -> str:
        return "agent" if state.get("claim") is not None else "manual_fallback"

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake)
    graph.add_node("agent", agent)
    graph.add_node("record_audit", record_audit)
    graph.add_node("tools", _tool_router(policy))
    graph.add_node("finalize", finalize)
    graph.add_node("manual_fallback", manual_fallback)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_after_intake, ["agent", "manual_fallback"])
    graph.add_edge("agent", "record_audit")
    graph.add_conditional_edges("record_audit", route_after_agent, ["tools", "finalize"])
    # Checks are single-shot and independent: the agent calls them in one batch, then finalize.
    graph.add_edge("tools", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("manual_fallback", END)
    return graph.compile()


def _tool_router(policy: PolicyStore):
    """A ToolNode that rebuilds the claim-bound tools from state at execution time."""

    def run_tools(state: AgentState) -> dict:
        tools = build_tools(state["claim"], policy)
        return ToolNode(tools).invoke(state)

    return run_tools


def evaluate_claim(raw_claim: dict, policy: Optional[PolicyStore] = None, model: Optional[str] = None) -> dict:
    """Run one claim through the agent and return decision + audit trail."""
    policy = policy or PolicyStore()
    graph = build_graph(policy, model)
    result = graph.invoke({"raw_claim": raw_claim})
    decision: ReimbursementDecision = result["decision"]
    return {
        "decision": decision.model_dump(),
        "audit_trail": result.get("audit", []),
    }
