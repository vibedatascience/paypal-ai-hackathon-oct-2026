"""Contextual bandit: disjoint linear Thompson sampling.

One Bayesian linear regression per action. At each decision we sample a weight
vector per action from its posterior and pick the action with the highest
sampled reward for this invoice's context.

Reward = 1 if the invoice is paid before the next epoch, minus a cost for every
reminder sent. The cost grows after the second reminder, so the bandit learns
to back off instead of spamming a client.
"""
from __future__ import annotations

import numpy as np

from .simulator import ACTIONS, EPOCH_DAYS, MAX_DAYS_OVERDUE, Invoice

FEATURES = [
    "bias",
    "is_business",
    "late_history",
    "reason_needs_po",
    "reason_forgot",
    "reason_cash_tight",
    "payment_run_soon",
    "days_overdue",
    "reminders_sent",
    "large_invoice",
]
REMINDER_COST = 0.03
FATIGUE_COST = 0.06  # extra cost per reminder already sent beyond 2


def context(inv: Invoice) -> np.ndarray:
    return np.array([
        1.0,
        float(inv.is_business),
        float(inv.has_late_history),
        float(inv.reason == "needs_po"),
        float(inv.reason == "forgot"),
        float(inv.reason == "cash_tight"),
        float(inv.days_to_next_run() < EPOCH_DAYS),
        inv.days_overdue / MAX_DAYS_OVERDUE,
        min(inv.reminders_sent, 5) / 5,
        float(inv.amount >= 2500),
    ])


def reward(action: str, paid: bool, reminders_before: int = 0) -> float:
    if action == "wait":
        return float(paid)
    return float(paid) - REMINDER_COST - FATIGUE_COST * max(0, reminders_before - 2)


class LinTS:
    def __init__(self, d: int = len(FEATURES), lam: float = 1.0, v: float = 0.35, seed: int = 0):
        self.d, self.v = d, v
        self.A = {a: lam * np.eye(d) for a in ACTIONS}
        self.b = {a: np.zeros(d) for a in ACTIONS}
        self.n = {a: 0 for a in ACTIONS}
        self.rng = np.random.default_rng(seed)

    def _posterior(self, a: str):
        A_inv = np.linalg.inv(self.A[a])
        return A_inv @ self.b[a], A_inv

    def choose(self, x: np.ndarray) -> str:
        best, best_val = None, -np.inf
        for a in ACTIONS:
            mu, cov = self._posterior(a)
            theta = self.rng.multivariate_normal(mu, self.v ** 2 * cov, method="cholesky")
            val = float(x @ theta)
            if val > best_val:
                best, best_val = a, val
        return best

    def update(self, x: np.ndarray, a: str, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
        self.n[a] += 1

    def expected(self, x: np.ndarray) -> dict:
        """Posterior mean and 90% interval of reward for each action."""
        out = {}
        for a in ACTIONS:
            mu, cov = self._posterior(a)
            m = float(x @ mu)
            s = float(np.sqrt(max(x @ (self.v ** 2 * cov) @ x, 0.0)))
            out[a] = {"mean": m, "lo": m - 1.645 * s, "hi": m + 1.645 * s, "n": self.n[a]}
        return out

    def to_dict(self) -> dict:
        return {"A": {a: self.A[a].tolist() for a in ACTIONS}, "b": {a: self.b[a].tolist() for a in ACTIONS},
                "n": self.n, "v": self.v}

    @classmethod
    def from_dict(cls, d: dict) -> "LinTS":
        m = cls(v=d.get("v", 0.35))
        for a in ACTIONS:
            m.A[a] = np.array(d["A"][a])
            m.b[a] = np.array(d["b"][a])
            m.n[a] = d["n"][a]
        return m
