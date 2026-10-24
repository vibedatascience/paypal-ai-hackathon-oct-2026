"""Policies compared in the experiment.

fixed_schedule : what most invoicing tools do. Friendly reminder on days 3, 12,
                 21, 30 and 45 after the due date.
llm_wrapper    : same schedule, but an LLM reads the client's reply and picks
                 the content (adds the PO when asked). Fixed timing, one tone.
bandit         : contextual Thompson sampling. Decides timing, format and tone
                 per invoice every epoch and learns from payments.
"""
from __future__ import annotations

SCHEDULE_DAYS = {3, 12, 21, 30, 45}


def fixed_schedule(inv) -> str:
    return "friendly" if inv.days_overdue in SCHEDULE_DAYS else "wait"


def llm_wrapper(inv) -> str:
    if inv.days_overdue not in SCHEDULE_DAYS:
        return "wait"
    if inv.reason == "needs_po":
        return "include_po"
    return "friendly"


def llm_wrapper_frequent(inv) -> str:
    """Ablation: the wrapper reminding every epoch. Shows frequency alone does not win."""
    return "include_po" if inv.reason == "needs_po" else "friendly"


POLICY_LABELS = {
    "fixed_schedule": "Fixed schedule",
    "llm_wrapper": "LLM wrapper",
    "llm_wrapper_frequent": "LLM wrapper, every 3 days",
    "bandit": "Learning bandit",
}
