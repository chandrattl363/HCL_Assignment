# Acme Corp — Travel & Expense Reimbursement Policy

_Mock policy for prototype use only. No real company data._

## 1. Eligible Expense Categories
The following categories are reimbursable when incurred for approved business travel:
- **airfare** — economy class only by default.
- **lodging** — hotel room charges (excludes minibar, spa, and in-room entertainment).
- **meals** — subject to a daily per-diem cap.
- **ground_transport** — taxi, rideshare, train, and rental car.
- **conference_fees** — registration for approved conferences and events.

## 2. Non-Reimbursable Categories
The following are **never reimbursable** and must be removed from any claim:
- **alcohol**
- **entertainment** (movies, events, in-room entertainment)
- **personal** (personal shopping, toiletries, laundry under 3 nights)
- **minibar**
- **spa**

## 3. Per-Diem and Category Limits (INR)
| Category          | Limit                         | Basis        |
|-------------------|-------------------------------|--------------|
| meals             | 7,500 per day                 | per day      |
| lodging           | 25,000 per night              | per night    |
| ground_transport  | 5,000 per day                 | per day      |
| airfare (economy) | 150,000 per trip              | per trip     |
| conference_fees   | 100,000 per trip              | per trip     |

Amounts above these caps are **deducted** (the within-cap portion is still approved). This
results in a **Partially Approve** decision.

## 4. Airfare Rules
- Economy class is reimbursable up to the airfare cap.
- **Business or first class requires prior manager approval** and must be routed to
  **Manual Review** if approval evidence is not attached.
- Economy tickets should be booked at least 7 days in advance where practical (advisory, not
  a hard rejection rule).

## 5. Receipt Requirements
- A receipt is **required for any single line item over ₹2,500**.
- Line items over ₹2,500 with no receipt attached are listed under **missing documents**.
- If a missing receipt is on a small item, deduct that item; if it is on a large/material
  item, route the claim to **Manual Review**.

## 6. Submission Window
- Claims must be submitted **within 30 days** of the travel end date.
- Claims submitted after 30 days are **Rejected** unless a documented exception applies.

## 7. Approval Matrix (by net reimbursable amount)
| Net amount (INR)      | Required approval          |
|-----------------------|----------------------------|
| up to 50,000          | Auto-approve               |
| 50,000.01 – 200,000   | Manager approval           |
| above 200,000         | Director approval (Manual Review) |

Claims whose net amount exceeds ₹200,000 are routed to **Manual Review** for director sign-off.

## 8. Duplicate Claims
- A claim that matches a previously submitted/paid claim (same employee, vendor, date, and
  amount) is a suspected duplicate and must be **Rejected**.

## 9. Decision Definitions
- **Approve** — fully compliant, within all limits, all receipts present, within auto-approve threshold.
- **Partially Approve** — compliant overall but with deductions for over-cap or ineligible items.
- **Reject** — duplicate, outside submission window, or entirely non-reimbursable.
- **Manual Review** — ambiguous, missing material receipts, business-class without approval, or net amount above the director threshold.
