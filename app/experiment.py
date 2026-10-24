"""Run the three policies on the same stream of simulated invoices.

All policies see identical invoices (same seeds), so differences come from the
policy, not from luck in which clients showed up.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict

import numpy as np

from .bandit import LinTS, context, reward
from .llm import classify_reply_fast
from .policies import POLICY_LABELS, fixed_schedule, llm_wrapper, llm_wrapper_frequent
from .simulator import ACTIONS, EPOCH_DAYS, MAX_DAYS_OVERDUE, SEGMENT_LABELS, SEGMENTS, make_invoice, step

PROFILES = [
    ("Asked for PO, AP run in 2 days",
     dict(is_business=True, has_late_history=False, reason="needs_po", run_soon=True, amount=5000)),
    ("Asked for PO, AP run in 10 days",
     dict(is_business=True, has_late_history=False, reason="needs_po", run_soon=False, amount=5000)),
    ("Business, said they forgot",
     dict(is_business=True, has_late_history=False, reason="forgot", run_soon=False, amount=1200)),
    ("Late history, no reply yet",
     dict(is_business=True, has_late_history=True, reason="unknown", run_soon=False, amount=1500)),
    ("Individual, no reply yet",
     dict(is_business=False, has_late_history=False, reason="unknown", run_soon=False, amount=400)),
]


def profile_context(p: dict, days_overdue: int = 6, reminders: int = 1) -> np.ndarray:
    return np.array([1.0, float(p["is_business"]), float(p["has_late_history"]),
                     float(p["reason"] == "needs_po"), float(p["reason"] == "forgot"),
                     float(p["reason"] == "cash_tight"), float(p["run_soon"]),
                     days_overdue / MAX_DAYS_OVERDUE, reminders / 5, float(p["amount"] >= 2500)])


def run(weeks: int = 26, per_week: int = 20, seed: int = 7) -> dict:
    n = weeks * per_week
    seeds = [seed * 100_000 + i for i in range(n)]
    start_epoch = [int((i // per_week) * 7 / EPOCH_DAYS) for i in range(n)]
    total_epochs = max(start_epoch) + MAX_DAYS_OVERDUE // EPOCH_DAYS + 1

    results, bandit_model, decisions = {}, None, []
    for name in POLICY_LABELS:
        invoices = [make_invoice(i, s) for i, s in enumerate(seeds)]
        model = LinTS(seed=seed) if name == "bandit" else None
        for t in range(total_epochs):
            for i, inv in enumerate(invoices):
                if start_epoch[i] > t or inv.closed:
                    continue
                before = inv.reminders_sent
                if name == "bandit":
                    x = context(inv)
                    a = model.choose(x)
                else:
                    a = {"fixed_schedule": fixed_schedule, "llm_wrapper": llm_wrapper,
                         "llm_wrapper_frequent": llm_wrapper_frequent}[name](inv)
                out = step(inv, a, classify_reply_fast)
                if name == "bandit":
                    model.update(x, a, reward(a, out["paid"], before))
                    decisions.append({"week": i // per_week, "segment": inv.segment, "action": a,
                                      "paid": out["paid"]})
        results[name] = invoices
        if name == "bandit":
            bandit_model = model

    return summarize(results, bandit_model, decisions, weeks, per_week)


def _days(inv) -> int:
    return inv.paid_day if inv.paid else MAX_DAYS_OVERDUE


def summarize(results, model, decisions, weeks, per_week) -> dict:
    kpis, weekly = {}, {}
    for name, invs in results.items():
        days = [_days(v) for v in invs]
        kpis[name] = {
            "label": POLICY_LABELS[name],
            "median_days_to_paid": statistics.median(days),
            "mean_days_to_paid": round(statistics.mean(days), 1),
            "paid_within_30d_pct": round(100 * sum(v.paid and v.paid_day <= 30 for v in invs) / len(invs), 1),
            "unpaid_at_60d_pct": round(100 * sum(not v.paid for v in invs) / len(invs), 1),
            "reminders_per_invoice": round(sum(v.reminders_sent for v in invs) / len(invs), 2),
        }
        # Rolling 3-week mean days-to-paid by issue week.
        by_week = defaultdict(list)
        for i, v in enumerate(invs):
            by_week[i // per_week].append(_days(v))
        series = []
        for w in range(weeks):
            window = [d for k in range(max(0, w - 2), w + 1) for d in by_week[k]]
            series.append(round(statistics.mean(window), 1))
        weekly[name] = series

    # What the bandit learned, by hidden segment, over the last 8 weeks.
    late = [d for d in decisions if d["week"] >= weeks - 8 and d["action"] != "wait"]
    by_seg = []
    for seg in SEGMENTS:
        acts = Counter(d["action"] for d in late if d["segment"] == seg)
        total = sum(acts.values()) or 1
        top = acts.most_common(1)[0][0] if acts else "wait"
        row = {"segment": SEGMENT_LABELS[seg], "top_action": top, "top_share_pct": round(100 * acts[top] / total, 1)}
        for a in ACTIONS[1:]:
            row[a] = round(100 * acts[a] / total, 1)
        by_seg.append(row)

    # Per-segment days-to-paid by policy.
    seg_table = []
    for seg in SEGMENTS:
        row = {"segment": SEGMENT_LABELS[seg]}
        for name, invs in results.items():
            ds = [_days(v) for v in invs if v.segment == seg]
            row[name] = round(statistics.mean(ds), 1) if ds else None
        seg_table.append(row)

    profiles = []
    for label, p in PROFILES:
        exp = model.expected(profile_context(p))
        best = max(exp, key=lambda a: exp[a]["mean"])
        profiles.append({"profile": label, "best_action": best,
                         **{a: round(exp[a]["mean"], 3) for a in ACTIONS},
                         **{f"{a}_lo": round(exp[a]["lo"], 3) for a in ACTIONS},
                         **{f"{a}_hi": round(exp[a]["hi"], 3) for a in ACTIONS}})

    return {"weeks": list(range(1, weeks + 1)), "weekly_mean_days": weekly, "kpis": kpis,
            "learned_by_segment": by_seg, "days_by_segment": seg_table, "profiles": profiles,
            "pulls": model.n, "model": model.to_dict(), "n_invoices": weeks * per_week}


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps(r["kpis"], indent=2))
    for row in r["days_by_segment"]:
        print(row)
    for row in r["learned_by_segment"]:
        print(row)
    for row in r["profiles"]:
        print(row["profile"], "->", row["best_action"])
    print({k: v[-1] for k, v in r["weekly_mean_days"].items()}, {k: v[2] for k, v in r["weekly_mean_days"].items()})
