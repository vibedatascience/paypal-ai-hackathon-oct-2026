"""Live lane: the trained bandit working on real PayPal sandbox invoices.

1. start(): create one PayPal invoice per demo client and send it.
2. step(): for every unpaid invoice, the bandit picks an action, the LLM writes
   the reminder, PayPal sends it (/remind). The simulated client responds. If
   the client pays, the payment is recorded in PayPal and the status is read
   back from PayPal. The bandit keeps learning from each outcome.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

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

    def _open(self, k: int, seg: str, email: str) -> dict:
        inv = make_invoice(k, _seed_for(seg, k))
        inv.client_name, company = DEMO_CLIENTS[k]
        inv.company = company if inv.is_business else None
        po = f"PO-{4400 + k}" if inv.company else None
        created = self.pp.create_invoice(client_name=inv.client_name, email=email, amount=inv.amount,
                                         po_number=po, item=ITEMS[seg],
                                         note="Thanks for your business. Pay with PayPal from this invoice.")
        iid = created.get("id") or created.get("href", "").rsplit("/", 1)[-1]
        self.pp.send_invoice(iid)
        status = self.pp.get_invoice(iid).get("status", "SENT")
        return {"inv": inv, "paypal_id": iid, "po": po, "status": status,
                "last_action": None, "subject": "", "note": "", "reason": "unknown"}

    def start(self) -> None:
        email = os.getenv("PAYPAL_RECIPIENT_EMAIL", "client@example.com")
        with ThreadPoolExecutor(max_workers=6) as ex:
            self.rows = list(ex.map(lambda ks: self._open(ks[0], ks[1], email), enumerate(DEMO_SEGMENTS)))

    def _choose(self, x: np.ndarray) -> str:
        exp = self.model.expected(x)  # greedy on the posterior mean for a clear demo
        return max(ACTIONS, key=lambda a: exp[a]["mean"])

    def _act(self, row: dict, a: str) -> tuple[dict, dict, str]:
        """Network work for one invoice: LLM writes, PayPal sends, client responds, PayPal records."""
        inv: Invoice = row["inv"]
        msg = llm.write_reminder(a, {"client_name": inv.client_name, "company": inv.company,
                                     "amount": inv.amount, "days_overdue": inv.days_overdue,
                                     "po_number": row["po"], "billing_contact": "ap@client.example",
                                     "last_reply": inv.last_reply})
        if a != "wait":
            self.pp.remind(row["paypal_id"], msg["subject"], msg["note"])
        out = step(inv, a, lambda t: llm.classify_reply(t)["reason"])
        if out["paid"]:
            self.pp.record_payment(row["paypal_id"], inv.amount)
        status = self.pp.get_invoice(row["paypal_id"]).get("status", row["status"])
        return msg, out, status

    def step(self) -> None:
        self.epoch += 1
        todo = []
        for row in self.rows:  # decisions first, from the current posterior
            inv: Invoice = row["inv"]
            if not inv.closed:
                x = context(inv)
                todo.append((row, x, inv.reminders_sent, self._choose(x)))
        # Each invoice's API calls are independent, so run them in parallel.
        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(lambda t: self._act(t[0], t[3]), todo))
        for (row, x, before, a), (msg, out, status) in zip(todo, results):
            inv = row["inv"]
            self.model.update(x, a, reward(a, out["paid"], before))
            row.update(status=status, last_action=a, subject=msg["subject"], note=msg["note"], reason=inv.reason)
            self.log.append({"epoch": self.epoch, "day": out["day"], "paypal_id": row["paypal_id"],
                             "client": inv.client_name, "action": ACTION_LABELS[a], "subject": msg["subject"],
                             "note": msg["note"], "writer": msg["source"], "reply": out["reply"] or "",
                             "reason": inv.reason, "paid": out["paid"], "paypal_status": status})

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
