"""Client simulator with hidden payment behaviors.

Each simulated client belongs to a hidden segment. The bandit never sees the
segment. It only sees observable context: business vs individual, late-payment
history, invoice size, days overdue, the calendar, and the reason the LLM
extracts from the client's reply (if the client replied).

Time moves in decision epochs of EPOCH_DAYS days. At each epoch the policy
picks one action per unpaid invoice. The client then pays (or not) before the
next epoch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

EPOCH_DAYS = 3
MAX_DAYS_OVERDUE = 60
PAYMENT_RUN_DAYS = (15, 30)  # corporate accounts payable runs

ACTIONS = ["wait", "friendly", "short_link", "include_po", "late_fee"]
ACTION_LABELS = {
    "wait": "Wait",
    "friendly": "Friendly email",
    "short_link": "Short note + 1-tap PayPal link",
    "include_po": "Resend with PO + billing details",
    "late_fee": "Firm note citing contract late fee",
}

SEGMENTS = ["corporate", "startup", "repeat_late", "individual"]
SEGMENT_WEIGHTS = [0.30, 0.30, 0.20, 0.20]
SEGMENT_LABELS = {
    "corporate": "Corporate (AP + PO)",
    "startup": "Startup founder",
    "repeat_late": "Repeat late payer",
    "individual": "Individual client",
}

# Probability of paying within one epoch, by hidden segment and action.
# Corporate is handled separately because it depends on PO + payment runs.
PAY_PROB = {
    "startup": {"wait": 0.05, "friendly": 0.30, "short_link": 0.60, "include_po": 0.25, "late_fee": 0.20},
    "repeat_late": {"wait": 0.03, "friendly": 0.05, "short_link": 0.08, "include_po": 0.05, "late_fee": 0.55},
    "individual": {"wait": 0.05, "friendly": 0.35, "short_link": 0.50, "include_po": 0.20, "late_fee": 0.15},
}
CORPORATE_NO_PO = {"wait": 0.02, "friendly": 0.04, "short_link": 0.03, "include_po": 0.0, "late_fee": 0.05}
CORPORATE_RUN_PAY = 0.90  # pays on a payment run once the PO is on file

# Probability the client replies to a reminder, and reply templates by reason.
REPLY_PROB = {"corporate": 0.60, "startup": 0.20, "repeat_late": 0.15, "individual": 0.25}
REPLIES = {
    "needs_po": [
        "Hi, our AP team can't process this without a PO number on the invoice. Can you resend?",
        "Accounts payable flagged this one: please add PO and our billing contact, then resubmit.",
        "Thanks. Finance needs the purchase order reference before it goes into the payment run.",
    ],
    "forgot": [
        "Oh no, this slipped my mind! Will take care of it.",
        "Sorry about that, totally forgot. Thanks for the nudge.",
        "Ah yes, apologies, I'll get to it shortly.",
    ],
    "cash_tight": [
        "Things are a bit tight this month, will pay as soon as I can.",
        "Cash flow is rough right now. Can we push this a couple of weeks?",
        "We're waiting on a client payment ourselves, should clear soon.",
    ],
}
SEGMENT_REASON = {"corporate": "needs_po", "startup": "forgot", "repeat_late": "cash_tight", "individual": "forgot"}

FIRST_NAMES = ["Ava", "Ben", "Chloe", "Dev", "Elena", "Femi", "Grace", "Hiro", "Isla", "Jon", "Kira", "Leo",
               "Maya", "Nina", "Omar", "Priya", "Quinn", "Ravi", "Sara", "Tom"]
COMPANIES = ["Acme Corp", "Northwind", "Globex", "Initech", "Umbrella Labs", "Stark Retail", "Wayne Media",
             "Hooli", "Vandelay Imports", "Pied Piper", "Soylent Foods", "Cyberdyne"]


@dataclass
class Invoice:
    id: str
    segment: str  # hidden
    client_name: str
    company: str | None
    amount: float
    is_business: bool
    has_late_history: bool
    start_day_of_month: int
    rng: random.Random
    days_overdue: int = 0
    reminders_sent: int = 0
    late_fee_sent: bool = False
    po_on_file: bool = False
    reason: str = "unknown"  # set by the classifier from a reply
    last_reply: str | None = None
    paid_day: int | None = None
    history: list = field(default_factory=list)

    @property
    def paid(self) -> bool:
        return self.paid_day is not None

    @property
    def closed(self) -> bool:
        return self.paid or self.days_overdue >= MAX_DAYS_OVERDUE

    def day_of_month(self, offset: int = 0) -> int:
        return (self.start_day_of_month + self.days_overdue + offset - 1) % 30 + 1

    def days_to_next_run(self) -> int:
        for k in range(0, 31):
            if self.day_of_month(k) in PAYMENT_RUN_DAYS:
                return k
        return 30


def make_invoice(idx: int, seed: int) -> Invoice:
    rng = random.Random(seed)
    segment = rng.choices(SEGMENTS, SEGMENT_WEIGHTS)[0]
    if segment == "corporate":
        is_business, amount = True, round(rng.uniform(2500, 9000), -1)
    elif segment == "startup":
        is_business, amount = True, round(rng.uniform(600, 2500), -1)
    elif segment == "repeat_late":
        is_business, amount = rng.random() < 0.5, round(rng.uniform(800, 3000), -1)
    else:
        is_business, amount = False, round(rng.uniform(150, 1200), -1)
    has_late_history = rng.random() < (0.8 if segment == "repeat_late" else 0.1)
    name = rng.choice(FIRST_NAMES)
    company = rng.choice(COMPANIES) if is_business else None
    return Invoice(
        id=f"INV-{idx:05d}", segment=segment, client_name=name, company=company, amount=amount,
        is_business=is_business, has_late_history=has_late_history,
        start_day_of_month=rng.randint(1, 30), rng=rng,
    )


def step(inv: Invoice, action: str, classify) -> dict:
    """Apply one action for one epoch. Returns the outcome."""
    rng = inv.rng
    seg = inv.segment
    sent = action != "wait"
    reply = None

    if sent:
        inv.reminders_sent += 1
        if action == "late_fee":
            inv.late_fee_sent = True
        if seg == "corporate" and action == "include_po":
            inv.po_on_file = True
        if rng.random() < REPLY_PROB[seg] and not (seg == "corporate" and inv.po_on_file):
            reply = rng.choice(REPLIES[SEGMENT_REASON[seg]])
            inv.last_reply = reply
            inv.reason = classify(reply)

    # Payment probability for this epoch.
    if seg == "corporate":
        run_this_epoch = inv.days_to_next_run() < EPOCH_DAYS
        p = CORPORATE_RUN_PAY if (inv.po_on_file and run_this_epoch) else CORPORATE_NO_PO[action]
    else:
        p = PAY_PROB[seg][action]
        if seg == "individual" and inv.late_fee_sent and action != "late_fee":
            p *= 0.7
    if inv.reminders_sent > 3:  # reminder fatigue
        p *= 0.75 ** (inv.reminders_sent - 3)

    paid = rng.random() < p
    if paid:
        inv.paid_day = inv.days_overdue + rng.randint(0, EPOCH_DAYS - 1)
    outcome = {"day": inv.days_overdue, "action": action, "paid": paid, "reply": reply}
    inv.history.append(outcome)
    inv.days_overdue += EPOCH_DAYS
    return outcome
