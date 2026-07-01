# Travel Reimbursement Approval Agent

Takes an employee travel reimbursement claim, checks it against a company policy, and gives
back one of four decisions: Approve, Partially Approve, Reject, or Manual Review.

I built it with LangGraph for the workflow and LangChain + Groq (via `init_chat_model`) for the
model, which is an open-source Llama 3.3 70B. Policy lookup uses FAISS with local HuggingFace
embeddings, and Pydantic handles the claim/decision schemas.

## How it works

There's a small LangGraph behind it:

```
claim (JSON) -> intake -> agent (LLM + tools) -> finalize -> validate -> decision (JSON)
```

Intake parses the raw claim into a Pydantic `Claim`. If it can't parse it, the claim goes
straight to Manual Review instead of erroring out. The agent step hands the claim and the tools
to the model and lets it decide which checks to run. Those checks (plus a policy lookup) run,
finalize asks the model for a structured decision, and validate recomputes the amounts with a
plain-Python rules engine and reconciles them with what the model said.

The thing I cared about most: the model orchestrates and writes the explanation, but it doesn't
touch the money math. Every amount and the final decision come out of `src/validator.py`, so a
payout can't ride on the model doing arithmetic correctly.

### Tools

| Tool | Type | What it checks |
|------|------|----------------|
| `policy_lookup` | retrieval | Semantic search over the policy (FAISS + embeddings) |
| `check_eligibility` | rule | Non-reimbursable or unknown categories |
| `check_spending_limits` | rule | Per-day / per-night / per-trip caps, and any overage |
| `check_receipt_completeness` | rule | Missing receipts for items over the threshold |
| `check_submission_window` | rule | 30-day submission deadline |
| `check_airfare_class` | rule | Business/first class needing manager approval |
| `detect_duplicate` | rule | Match against previously paid claims |
| `check_approval_threshold` | rule | Approval level from the approval matrix |

### Output

Every claim returns this shape (defined in `src/models.py`):

```json
{
  "claim_id": "CLM-002",
  "decision": "Partially Approve",
  "claimed_amount": 137000.0,
  "approved_amount": 122500.0,
  "rejected_amount": 14500.0,
  "deductions": [{"item": "Hotel, 2 nights", "amount": 10000.0, "reason": "over cap (50000)"}],
  "missing_documents": [],
  "policy_references": ["Policy Section 3: Per-Diem and Category Limits"],
  "required_approval": "Manager approval",
  "confidence": 0.9,
  "reason_codes": ["OVER_CAP"],
  "explanation": "..."
}
```

## Setup

Needs Python 3.11+ (I used 3.12) and uv (https://docs.astral.sh/uv/).

```bash
uv sync                       # creates .venv and installs everything from pyproject.toml
cp .env.example .env          # then paste your Groq key into .env
```

Grab a free Groq key at https://console.groq.com/keys. Groq runs open-source models (Llama 3.3,
Qwen, GPT-OSS) on a free tier that has been more than enough for this. Prefix commands with
`uv run` to use the environment, e.g. `uv run python run.py --all`.

| Variable | Default | Notes |
|----------|---------|-------|
| `GROQ_API_KEY` | required | Free Groq API key |
| `LLM_MODEL` | `groq:llama-3.3-70b-versatile` | Provider-qualified model name |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |

The embedding model (about 80 MB) downloads once on the first run. Retrieval itself needs no API
key; only the agent's reasoning calls Groq. If the model is down or rate-limited, the run still
finishes on the deterministic path and the result is tagged `LLM_UNAVAILABLE`.

## Running it

```bash
uv run python run.py --all                 # all five sample claims, table output
uv run python run.py --id CLM-002          # one claim
uv run python run.py --file claim.json     # your own claim file (object or list)
uv run python run.py --all --save outputs  # also write JSON to outputs/
```

### As an API

```bash
uv run uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/docs for the Swagger UI, or hit it directly:

```bash
curl -X POST http://127.0.0.1:8000/evaluate/sample/CLM-002
```

| Method and path | Purpose |
|-----------------|---------|
| `GET /health` | Liveness and which retriever is active |
| `GET /sample-claims` | The bundled sample claims |
| `POST /evaluate` | Evaluate a claim (validated against the `Claim` schema; bad input returns 422) |
| `POST /evaluate/sample/{claim_id}` | Evaluate a bundled sample by id |

## Sample claims

The five bundled claims cover each outcome:

| Claim | Scenario | Decision |
|-------|----------|----------|
| CLM-001 | Clean 1-day conference, all within caps, under ₹50,000 | Approve |
| CLM-002 | Hotel and meals over the per-diem caps | Partially Approve (deduct ₹14,500) |
| CLM-003 | Submitted 70 days after travel (window is 30) | Reject |
| CLM-004 | Business-class airfare, missing ₹60,000 receipt, net over ₹200,000 | Manual Review |
| CLM-005 | Duplicate of an already-paid claim | Reject |

Generated outputs for all five are in [`outputs/`](outputs/).

## Tests

```bash
uv run pytest tests/test_validator.py -q      # offline, no API key needed
uv run pytest tests/test_agent_eval.py -q -s  # live agent, needs GROQ_API_KEY
```

The offline tests are the important ones. They pin the expected decision and amount for each
sample claim straight from the rules engine, so correctness doesn't depend on the model.

## Some decisions I made

The model picks tools and writes the explanation, but the numbers and the label come from the
rules engine. I didn't want an LLM quietly changing a reimbursement amount.

The validator also acts as the fallback. If the model's label disagrees with the rules engine,
I keep the rules engine's answer, tag it `VALIDATOR_OVERRIDE`, and drop the confidence. Same
thing if the Groq call fails or gets rate-limited, the run finishes on the deterministic path.

For the policy lookup I load the markdown with LangChain's `TextLoader`, split it by section,
embed it with a local sentence-transformers model, and index it in FAISS. If the embeddings
can't load for some reason it falls back to a keyword search so it still runs.

Pydantic does the input validation (a bad claim becomes Manual Review) and keeps the output
schema consistent no matter how the model phrases things. And each run keeps a list of which
tools were called, which was handy while debugging and makes the decisions auditable.

## Assumptions and limitations

Assumptions:
- Everything is mock data. No real employee or company information.
- Single currency (INR). Line items are already totals.
- "Net amount" for the approval matrix means the approved amount after deductions.
- An exact match (same employee, vendor, date, amount) against a paid claim is a duplicate and
  gets rejected.

Simplifications:
- The paid-claim history for duplicate detection is a small JSON file, not a database.
- Per-diem caps are flat, with no city or grade tiers.
- The FAISS index is rebuilt in memory each run, which is fine for a policy this small.
- The graph is compiled per claim because the tools are bound to the claim.

If I had more time:
- Persist the FAISS index and use a larger, multi-file policy.
- Currency conversion and per-region policies.
- A real datastore for claim history with fuzzy duplicate matching.
- Route approvals to the right person for the Manager and Director tiers.
- A small UI on top of the API.

## Layout

```
data/
  travel_policy.md      mock policy (loaded, chunked, embedded)
  limits.json           caps, approval matrix, category lists
  sample_claims.json    5 sample claims
  claims_history.json   paid claims for duplicate detection
src/
  models.py             Pydantic schemas (Claim, ReimbursementDecision, ...)
  policy_store.py       policy loading + FAISS retrieval
  tools.py              the checks + LangChain tool wrappers
  validator.py          deterministic rules engine
  agent.py              LangGraph workflow
  cli.py                CLI runner
  api.py                FastAPI endpoints
tests/
  test_validator.py     offline rules tests
  test_agent_eval.py    live agent test (needs GROQ_API_KEY)
run.py                  entry point
pyproject.toml          dependencies (uv sync)
```
