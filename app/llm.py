"""LLM layer: read client replies and write reminders.

The LLM does two jobs:
1. classify_reply: turn a free-text client reply into a reason label that the
   bandit uses as a context feature (needs_po, forgot, cash_tight, other).
2. write_reminder: write the reminder in the format and tone the bandit chose.
   The text is sent to the client through PayPal's invoice reminder API.

Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) to use a real model. Without a key,
deterministic fallbacks keep the app runnable. The offline simulation uses the
fast keyword classifier so thousands of replies run in seconds.
"""
from __future__ import annotations

import json
import os

import httpx

REASONS = ["needs_po", "forgot", "cash_tight", "other"]

_KEYWORDS = {
    "needs_po": ["po ", "po,", "po.", "purchase order", "accounts payable", "ap team", "finance needs", "billing contact"],
    "cash_tight": ["tight", "cash flow", "push this", "waiting on a client", "as soon as i can"],
    "forgot": ["forgot", "slipped", "apolog", "sorry", "nudge"],
}


def classify_reply_fast(text: str) -> str:
    t = f" {text.lower()} "
    for reason in ("needs_po", "cash_tight", "forgot"):
        if any(k in t for k in _KEYWORDS[reason]):
            return reason
    return "other"


def provider() -> str | None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def _chat(system: str, user: str, max_tokens: int = 400) -> str:
    p = provider()
    if p == "anthropic":
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"), "max_tokens": max_tokens,
                  "system": system, "messages": [{"role": "user", "content": user}]},
            timeout=30,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    if p == "openai":
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("no LLM provider configured")


def classify_reply(text: str) -> dict:
    """Return {"reason": ..., "source": "llm"|"fallback"}."""
    if not provider():
        return {"reason": classify_reply_fast(text), "source": "fallback"}
    system = ("You label why a client has not paid an invoice, based on their reply. Answer with JSON only: "
              '{"reason": one of ["needs_po","forgot","cash_tight","other"]}. '
              "needs_po = their accounts payable needs a PO number, billing contact or other paperwork. "
              "forgot = they overlooked it. cash_tight = they lack funds right now. other = anything else.")
    try:
        raw = _chat(system, text, max_tokens=50)
        reason = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])["reason"]
        if reason in REASONS:
            return {"reason": reason, "source": "llm"}
    except Exception:
        pass
    return {"reason": classify_reply_fast(text), "source": "fallback"}


STYLE = {
    "friendly": "A warm, polite email of 3-4 sentences. Mention the amount and that the invoice is past due.",
    "short_link": "Two short sentences max, written to be read on a phone. Amount first. Point to the PayPal "
                  "pay button on the invoice. No pleasantries.",
    "include_po": "A brief note for an accounts payable team. Restate the PO number, billing contact, amount and "
                  "due date on separate lines so AP can process it in the next payment run.",
    "late_fee": "Firm but professional. State the amount, the days overdue, and that the late fee in the signed "
                "agreement applies if unpaid in 5 days. 3 sentences.",
}


def _fallback_reminder(action: str, inv: dict) -> dict:
    amt = f"${inv['amount']:,.2f}"
    name = inv["client_name"]
    days = inv["days_overdue"]
    po = inv.get("po_number", "PO-4471")
    if action == "short_link":
        return {"subject": f"{amt} due: pay in one tap",
                "note": f"{amt} is {days} days past due. Tap Pay Now on this invoice to settle it in PayPal."}
    if action == "include_po":
        return {"subject": f"Invoice resubmitted with {po}",
                "note": f"Hi {name}, resubmitting for your AP team.\nPO: {po}\nBilling contact: "
                        f"{inv.get('billing_contact', 'ap@client.example')}\nAmount: {amt}\nDue: {days} days ago"}
    if action == "late_fee":
        return {"subject": f"Overdue {days} days: late fee applies in 5 days",
                "note": f"Hi {name}, this invoice for {amt} is {days} days overdue. Per our agreement, the late "
                        f"fee applies if it is not paid within 5 days. Pay Now is on the invoice."}
    return {"subject": f"Friendly reminder: {amt} invoice",
            "note": f"Hi {name}, hope all is well. A quick reminder that the invoice for {amt} is now {days} days "
                    f"past due. You can pay it directly from this PayPal invoice. Thanks!"}


def write_reminder(action: str, inv: dict) -> dict:
    """Return {"subject", "note", "source"} for a reminder in the chosen style."""
    if action == "wait":
        return {"subject": "", "note": "", "source": "none"}
    if not provider():
        return {**_fallback_reminder(action, inv), "source": "fallback"}
    system = ("You write invoice payment reminders on behalf of a freelancer. Follow the requested style "
              "exactly. Output JSON only: {\"subject\": str (<= 60 chars), \"note\": str (<= 600 chars)}. "
              "Do not invent amounts, dates or fees beyond what is given. Plain text, no markdown.")
    user = json.dumps({"style": STYLE[action], "client_name": inv["client_name"], "company": inv.get("company"),
                       "amount_usd": inv["amount"], "days_overdue": inv["days_overdue"],
                       "po_number": inv.get("po_number"), "billing_contact": inv.get("billing_contact"),
                       "client_last_reply": inv.get("last_reply")})
    try:
        raw = _chat(system, user)
        out = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {"subject": out["subject"][:60], "note": out["note"][:600], "source": "llm"}
    except Exception:
        return {**_fallback_reminder(action, inv), "source": "fallback"}
