"""The self-optimizing loop: watch → suggest → test → keep.

Each segment (e.g. "mobile, big cart, new shopper") has a current checkout
layout (its "default"). New ideas are tested against it with Thompson sampling
on a Beta-Bernoulli model per segment and layout: every shopper is shown the
layout whose sampled conversion rate is highest, so traffic moves to what works
while uncertain ideas still get tried.

Guardrails:
  * a test is stopped early once it is very likely (97%) worse than the default
  * an idea becomes the new default only if it is very likely (95%) the best
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import sim

MC = 4000  # Monte Carlo draws for posterior comparisons
WINDOW = 250  # sessions per point on the timeline


@dataclass
class Variant:
    vid: str
    name: str
    config: dict
    hypothesis: str
    source: str  # original | claude | rules
    round: int
    targets: list
    stats: dict = field(default_factory=dict)  # seg -> [paid, n]
    status: dict = field(default_factory=dict)  # seg -> testing | kept | stopped | retired | default
    note: dict = field(default_factory=dict)  # seg -> why stopped / kept

    def ab(self, seg):
        paid, n = self.stats.get(seg, [0, 0])
        return 1 + paid, 1 + n - paid


class Optimizer:
    def __init__(self, seed: int = 11, goal: str = "orders"):
        self.goal = goal  # orders: any completed checkout · branded: paid with PayPal, Venmo or Pay Later
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)
        self.sid = 0
        self.round = 0
        self.variants: dict[str, Variant] = {}
        orig = Variant("v0", "Original layout", dict(sim.ORIGINAL), "PayPal's default button stack.", "original", 0,
                       list(sim.SEGMENTS))
        for s in sim.SEGMENTS:
            orig.status[s] = "default"
        self.variants["v0"] = orig
        self.default = {s: "v0" for s in sim.SEGMENTS}
        self.log: list[dict] = []
        self.timeline: list[dict] = []
        self.funnels: list[dict] = []
        self.changelog: list[dict] = []
        self._win = defaultdict(float)
        self.fun = defaultdict(Counter)  # (vid, seg) -> funnel counts, cumulative
        self.phase = "new"

    # ---------- running sessions ----------
    def _candidates(self, seg):
        out = [self.default[seg]]
        out += [v.vid for v in self.variants.values() if v.status.get(seg) == "testing"]
        return out

    def _pick(self, seg):
        c = self._candidates(seg)
        if len(c) == 1:
            return c[0]
        draws = [self.np.beta(*self.variants[v].ab(seg)) for v in c]
        return c[int(np.argmax(draws))]

    def run(self, n: int, phase: str):
        self.phase = phase
        for _ in range(n):
            self.sid += 1
            sh = sim.make_shopper(self.sid, self.rng)
            seg = sh.segment
            vid = self._pick(seg)
            v = self.variants[vid]
            out = sim.run_session(sh, v.config, self.rng)
            st = v.stats.setdefault(seg, [0, 0])
            st[1] += 1
            win = out["paid"] and (self.goal == "orders" or out["method"] in sim.BRANDED)
            st[0] += int(win)
            # funnel counts per layout and segment (the advisor reads the current default's)
            if True:
                f = self.fun[(vid, seg)]
                f["sessions"] += 1
                if out["drop"] == "buttons":
                    f["left_at_buttons"] += 1
                if out["method"]:
                    pos = v.config["order"].index(out["method"]) + 1
                    f[f"chose_{out['method']}"] += 1
                    f[f"pos_{out['method']}"] = pos
                    f[f"done_{out['method']}"] += int(out["paid"])
                f["paid"] += int(out["paid"])
            # timeline: expected conversion of what we served vs the original layout, same shopper
            served, served_b = sim.expected_conversion(sh, v.config)
            orig, orig_b = sim.expected_conversion(sh, sim.ORIGINAL)
            w = self._win
            w["n"] += 1; w["served"] += served; w["orig"] += orig; w["served_b"] += served_b; w["orig_b"] += orig_b
            w["paid"] += int(out["paid"]); w["tests"] += int(vid != self.default[seg])
            if w["n"] >= WINDOW:
                self._flush()
            if len(self.log) < 4000 or self.sid % 5 == 0:
                self.log.append({"session": self.sid, "round": self.round, "segment": sim.seg_label(seg),
                                 "device": sh.device, "amount": sh.amount, "layout": v.name, "vid": vid,
                                 "steps": " → ".join(f"{a}:{b}" for a, b in out["events"][1:]),
                                 "paid": out["paid"], "method": sim.METHOD_LABEL.get(out["method"] or "", "")})
                self.log = self.log[-4000:]
            if self.sid % 250 == 0:
                self._guardrail()
        self.funnels.append({"round": self.round, "phase": phase, "by_segment": self._funnel_summary()})

    def _flush(self):
        w = self._win
        n = w["n"]
        self.timeline.append({"session": self.sid, "round": self.round, "phase": self.phase,
                              "served": round(100 * w["served"] / n, 2), "original": round(100 * w["orig"] / n, 2),
                              "branded": round(100 * w["served_b"] / n, 2), "branded_original": round(100 * w["orig_b"] / n, 2),
                              "realized": round(100 * w["paid"] / n, 2), "test_share": round(100 * w["tests"] / n, 1)})
        self._win = defaultdict(float)

    def _funnel_summary(self):
        out = {}
        for s in sim.SEGMENTS:
            f = self.fun[(self.default[s], s)]
            n = f["sessions"] or 1
            cfg = self.variants[self.default[s]].config
            out[s] = {"label": sim.seg_label(s), "layout": self.variants[self.default[s]].name,
                      "button_order": cfg["order"], "paylater_msg": cfg["paylater_msg"], "sessions": f["sessions"],
                      "conversion_pct": round(100 * f["paid"] / n, 1),
                      "left_without_clicking_pct": round(100 * f["left_at_buttons"] / n, 1),
                      "methods": {m: {"position": cfg["order"].index(m) + 1,
                                      "chosen_pct": round(100 * f[f"chose_{m}"] / n, 1),
                                      "completed_pct_of_chosen": round(100 * f[f"done_{m}"] / max(f[f"chose_{m}"], 1), 1)}
                                  for m in cfg["order"]}}
        return out

    # ---------- statistics ----------
    def _draws(self, vid, seg):
        return self.np.beta(*self.variants[vid].ab(seg), size=MC)

    def _guardrail(self):
        for v in self.variants.values():
            for seg, st in list(v.status.items()):
                if st != "testing" or v.stats.get(seg, [0, 0])[1] < 120:
                    continue
                d = self.default[seg]
                p_worse = float(np.mean(self._draws(v.vid, seg) < self._draws(d, seg)))
                if p_worse > 0.97:
                    v.status[seg] = "stopped"
                    v.note[seg] = f"Stopped early: {p_worse:.0%} likely worse than the current layout"
                    self._change("stopped", v, seg, d)

    MAX_N = 2500  # a test that has not decided after this many shoppers is called a tie

    def finalize(self):
        """End of an experiment round: keep clear winners; undecided tests keep running unless they hit MAX_N."""
        for seg in sim.SEGMENTS:
            cands = self._candidates(seg)
            if len(cands) == 1:
                continue
            draws = np.vstack([self._draws(c, seg) for c in cands])
            p_best = np.bincount(np.argmax(draws, axis=0), minlength=len(cands)) / MC
            best = int(np.argmax(p_best))
            old = self.default[seg]
            for i, c in enumerate(cands):
                v = self.variants[c]
                if c == old:
                    continue
                if i == best and p_best[i] >= 0.95:
                    v.status[seg] = "kept"
                    v.note[seg] = f"Kept: {p_best[i]:.0%} likely the best layout"
                    self.variants[old].status[seg] = "retired" if old != "v0" else "replaced"
                    self.default[seg] = c
                    self._change("kept", v, seg, old)
                elif p_best[i] < 0.02 or v.stats.get(seg, [0, 0])[1] >= self.MAX_N:
                    v.status[seg] = "stopped"
                    if p_best[i] < 0.02:
                        worse = self.rate(c, seg) < self.rate(old, seg)
                        v.note[seg] = ("Stopped: worse than the current layout" if worse
                                       else "Beaten by a better idea in the same test")
                        self._change("stopped" if worse else "lost", v, seg, old)
                    else:
                        v.note[seg] = f"No clear win after {self.MAX_N:,} shoppers"
                        self._change("no_win", v, seg, old)

        if self.funnels:
            self.funnels.append({"round": self.round, "phase": "decided", "by_segment": self._funnel_summary()})

    def rate(self, vid, seg):
        a, b = self.variants[vid].ab(seg)
        return a / (a + b)

    def _change(self, kind, v, seg, old):
        new_r, old_r = self.rate(v.vid, seg), self.rate(old, seg)
        self.changelog.append({"round": self.round, "kind": kind, "segment": sim.seg_label(seg), "seg": seg,
                               "variant": v.name, "vid": v.vid, "hypothesis": v.hypothesis,
                               "new_pct": round(100 * new_r, 1), "old_pct": round(100 * old_r, 1),
                               "old_layout": self.variants[old].name, "n": v.stats.get(seg, [0, 0])[1]})

    # ---------- adding ideas ----------
    def add_variant(self, name, config, hypothesis, targets, source):
        ok, why = sim.valid_config(config)
        if not ok:
            return None, why
        segs = sim.SEGMENTS if targets in ("all", ["all"]) else [t for t in targets if t in sim.SEGMENTS]
        segs = [s for s in segs if not self._tested(config, s)]
        if not segs:
            return None, "already tested for these shoppers"
        vid = f"v{len(self.variants)}"
        v = Variant(vid, name[:60], config, hypothesis[:160], source, self.round, segs)
        for s in segs:
            v.status[s] = "testing"
        self.variants[vid] = v
        return v, ""

    def _tested(self, cfg, seg):
        key = (tuple(cfg["order"]), cfg["paylater_msg"])
        return any((tuple(v.config["order"]), v.config["paylater_msg"]) == key and seg in v.status
                   for v in self.variants.values())

    # ---------- view ----------
    def state(self) -> dict:
        segs = []
        for s in sim.SEGMENTS:
            d = self.variants[self.default[s]]
            segs.append({"seg": s, "label": sim.seg_label(s), "default": d.vid, "default_name": d.name,
                         "config": d.config, "original": sim.ORIGINAL,
                         "rate": round(100 * self.rate(d.vid, s), 1), "orig_rate": round(100 * self.rate("v0", s), 1),
                         "n": sum(v.stats.get(s, [0, 0])[1] for v in self.variants.values())})
        vs = []
        for v in self.variants.values():
            rows = []
            for s, st in v.status.items():
                paid, n = v.stats.get(s, [0, 0])
                a, b = v.ab(s)
                lo, hi = np.quantile(self.np.beta(a, b, size=2000), [0.05, 0.95])
                drow = self.variants["v0"].stats.get(s, [0, 0])
                rows.append({"seg": s, "label": sim.seg_label(s), "status": st, "n": n, "paid": paid,
                             "rate": round(100 * a / (a + b), 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1),
                             "orig_rate": round(100 * (1 + drow[0]) / (2 + drow[1]), 1), "note": v.note.get(s, "")})
            vs.append({"vid": v.vid, "name": v.name, "config": v.config, "hypothesis": v.hypothesis,
                       "source": v.source, "round": v.round, "segments": rows})
        tl = self.timeline
        first = [t for t in tl if t["phase"] == "baseline"]
        last = tl[-8:] if tl else []
        avg = lambda xs, k: round(sum(x[k] for x in xs) / len(xs), 1) if xs else None
        kpi = {"goal": self.goal, "start": avg(first, "served"), "now": avg(last, "served"), "orig_now": avg(last, "original"),
               "branded_start": avg(first, "branded"), "branded_now": avg(last, "branded"),
               "sessions": self.sid, "round": self.round,
               "ideas": len(self.variants) - 1,
               "kept": sum(1 for c in self.changelog if c["kind"] == "kept"),
               "stopped": sum(1 for c in self.changelog if c["kind"] != "kept")}
        return {"segments": segs, "variants": vs, "timeline": tl, "funnels": self.funnels[-6:],
                "changelog": self.changelog[::-1], "log": self.log[-300:][::-1], "kpi": kpi,
                "labels": {"methods": sim.METHOD_LABEL}}
