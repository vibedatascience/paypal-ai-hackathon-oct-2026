"""Live lane: the trained bandit working on real PayPal sandbox invoices.

1. start(): create one PayPal invoice per demo client and send it.
2. step(): for every unpaid invoice, the bandit picks an action, the LLM writes
   the reminder, PayPal sends it (/remind). The simulated client responds. If
   the client pays, the payment is recorded in PayPal and the status is read
   back from PayPal. The bandit keeps learning from each outcome.
"""
from __future__ import annotations

import os

import numpy as np

from . import llm, paypal
from .bandit import LinTS, context, reward
from .simulator import ACTION_LABELS, ACTIONS, SEGMENT_LABELS, Invoice, make_invoice, step

# Hand-picked seeds so the demo shows one client from each hidden segment.
DEMO_SEGMENTS = ["corporate", "startup", "repeat_late", "individual", "corporate", "startup"]
ITEMS = {"corporate": "Brand identity system", "startup": "Landing page design",
         "repeat_late": "Monthly retainer", "individual": "Event photography"}


DEMO_CLIENTS = [("Jordan", "Acme Corp"), ("Maya", "Pied Piper"), ("Tom", None), ("Priya", None),
                ("Hiro", "Globex"), ("Leo", "Hooli")]


def _seed_for(segment: str, k: int) -> int:
    """First seed whose invoice has this segment and a typical late-payment history for it."""
    s = 900_000 + k * 1000
    while True:
        inv = make_invoice(0, s)
        if inv.segment == segment and inv.has_late_history == (segment == "repeat_late"):
            return s
        s += 1


class LiveLane:
    def __init__(self, model: LinTS | None):
        self.model = model or LinTS()
        self.pp = paypal.client()
        self.rows: list[dict] = []
        self.log: list[dict] = []
        self.epoch = 0

    @property
    def mode(self) -> str:
        return self.pp.mode

    def start(self) -> None:
        email = os.getenv("PAYPAL_RECIPIENT_EMAIL", "client@example.com")
        for k, seg in enumerate(DEMO_SEGMENTS):
            inv = make_invoice(k, _seed_for(seg, k))
            inv.client_name, company = DEMO_CLIENTS[k]
            inv.company = company if inv.is_business else None
            po = f"PO-{4400 + k}" if inv.is_business else None
            created = self.pp.create_invoice(client_name=inv.client_name, email=email, amount=inv.amount,
                                             po_number=po, item=ITEMS[seg],
                                             note="Thanks for your business. Pay with PayPal from this invoice.")
            iid = created.get("id") or created.get("href", "").rsplit("/", 1)[-1]
            self.pp.send_invoice(iid)
            status = self.pp.get_invoice(iid).get("status", "SENT")
            self.rows.append({"inv": inv, "paypal_id": iid, "po": po, "status": status,
                              "last_action": None, "subject": "", "note": "", "reason": "unknown"})

    def _choose(self, x: np.ndarray) -> str:
        exp = self.model.expected(x)  # greedy on the posterior mean for a clear demo
        return max(ACTIONS, key=lambda a: exp[a]["mean"])

    def step(self) -> None:
        self.epoch += 1
        for row in self.rows:
            inv: Invoice = row["inv"]
            if inv.closed:
                continue
            x = context(inv)
            before = inv.reminders_sent
            a = self._choose(x)
            msg = llm.write_reminder(a, {"client_name": inv.client_name, "company": inv.company,
                                         "amount": inv.amount, "days_overdue": inv.days_overdue,
                                         "po_number": row["po"], "billing_contact": "ap@client.example",
                                         "last_reply": inv.last_reply})
            if a != "wait":
                self.pp.remind(row["paypal_id"], msg["subject"], msg["note"])
            out = step(inv, a, lambda t: llm.classify_reply(t)["reason"])
            self.model.update(x, a, reward(a, out["paid"], before))
            if out["paid"]:
                self.pp.record_payment(row["paypal_id"], inv.amount)
            row.update(status=self.pp.get_invoice(row["paypal_id"]).get("status", row["status"]),
                       last_action=a, subject=msg["subject"], note=msg["note"], reason=inv.reason)
            self.log.append({"epoch": self.epoch, "day": out["day"], "paypal_id": row["paypal_id"],
                             "client": inv.client_name, "action": ACTION_LABELS[a], "subject": msg["subject"],
                             "note": msg["note"], "writer": msg["source"], "reply": out["reply"] or "",
                             "reason": inv.reason, "paid": out["paid"], "paypal_status": row["status"]})

    def table(self) -> list[dict]:
        out = []
        for row in self.rows:
            inv = row["inv"]
            out.append({"paypal_id": row["paypal_id"], "client": inv.client_name, "company": inv.company or "",
                        "segment_hidden": SEGMENT_LABELS[inv.segment], "amount": inv.amount,
                        "days_overdue": inv.days_overdue, "reminders": inv.reminders_sent,
                        "llm_reason": inv.reason,
                        "last_action": ACTION_LABELS.get(row["last_action"], "") if row["last_action"] else "",
                        "paypal_status": row["status"],
                        "paid_day": inv.paid_day if inv.paid else None})
        return out
