"""The 'suggest' step: Claude reads the checkout funnel and proposes layouts to test.

Claude only proposes. It can change three settings that a merchant controls
through the PayPal JavaScript SDK (button order, Pay Later message position,
headline). Every proposal is validated and then has to win an experiment
before any shopper sees it by default.
"""
from __future__ import annotations

import json

from .. import llm
from . import sim

SYSTEM = """You are a checkout UX analyst for an online store that uses PayPal's payment buttons.
You get funnel data per shopper segment for the layout each segment currently sees.
You may only change these settings:
  order: list of payment buttons top to bottom, from ["paypal","venmo","paylater","card"];
         2 to 4 items, no repeats, must include "paypal" and "card"
  paylater_msg: where the "Pay in 4" message shows: "none", "above" (above the buttons) or "below" (below the total)
  headline: text above the buttons, 48 characters max
Find the biggest drop-offs and propose up to 3 experiments that could fix them.
Rules: target the segments where the problem is (use the exact segment keys), do not repeat a layout listed in
already_tested for the same segment, and write the hypothesis as one plain sentence a shop owner understands
(20 words max, no jargon).
Answer with JSON only:
{"experiments":[{"name":"<=5 words","target_segments":["<segment key>",...],
  "config":{"order":[...],"paylater_msg":"...","headline":"..."},"hypothesis":"..."}]}"""


def _tested(opt) -> dict:
    out = {}
    for v in opt.variants.values():
        for s in v.status:
            out.setdefault(s, []).append({"order": v.config["order"], "paylater_msg": v.config["paylater_msg"],
                                          "result": v.status[s]})
    return out


def propose(opt, max_ideas: int = 3) -> dict:
    funnel = opt.funnels[-1]["by_segment"] if opt.funnels else {}
    payload = {"segments": funnel, "already_tested": _tested(opt)}
    ideas, source, raw = [], "rules", ""
    if llm.provider():
        try:
            raw = llm._chat(SYSTEM, json.dumps(payload), max_tokens=1400)
            ideas = json.loads(raw[raw.find("{"): raw.rfind("}") + 1], strict=False)["experiments"]
            source = "claude"
        except Exception:
            ideas = []
    if not ideas:
        ideas, source = rule_ideas(funnel), "rules"
    added, rejected = [], []
    for it in ideas:
        if len(added) >= max_ideas:
            break
        cfg = {"order": it.get("config", {}).get("order"), "paylater_msg": it.get("config", {}).get("paylater_msg", "none"),
               "headline": it.get("config", {}).get("headline", "Choose a way to pay")}
        v, why = opt.add_variant(it.get("name", "Idea"), cfg, it.get("hypothesis", ""), it.get("target_segments", ["all"]), source)
        (added if v else rejected).append({"name": it.get("name"), "why": why} if not v else v.vid)
    if not added and source == "claude":  # Claude repeated itself: fall back to rules so the loop keeps moving
        for it in rule_ideas(funnel):
            v, why = opt.add_variant(it["name"], it["config"], it["hypothesis"], it["target_segments"], "rules")
            if v:
                added.append(v.vid)
            if len(added) >= max_ideas:
                break
    return {"source": source, "added": added, "rejected": rejected}


def rule_ideas(funnel: dict) -> list:
    """Deterministic fallback when no LLM key is set. Reads the same funnel numbers."""
    ideas = []
    mob = [s for s in funnel if s.startswith("mobile")]
    big = [s for s in funnel if "|big|" in s]
    new_desk = [s for s in funnel if s.startswith("desktop") and s.endswith("new")]
    ideas.append({"name": "Venmo first on phones", "target_segments": mob,
                  "config": {"order": ["venmo", "paypal", "card"], "paylater_msg": "none", "headline": "Pay in one tap"},
                  "hypothesis": "Phone shoppers drop at the PayPal login; Venmo approves in one tap."})
    ideas.append({"name": "Pay in 4 up top", "target_segments": big,
                  "config": {"order": ["paylater", "paypal", "card"], "paylater_msg": "above", "headline": "Split it into 4 payments"},
                  "hypothesis": "Big carts stall at the total; showing 4 smaller payments first should help."})
    ideas.append({"name": "Card first for new shoppers", "target_segments": new_desk,
                  "config": {"order": ["card", "paypal", "venmo"], "paylater_msg": "none", "headline": "Pay with card or PayPal"},
                  "hypothesis": "Most new shoppers have no PayPal account, so a card option should come first."})
    ideas.append({"name": "Fewer buttons", "target_segments": ["all"],
                  "config": {"order": ["paypal", "card"], "paylater_msg": "none", "headline": "Checkout"},
                  "hypothesis": "Four buttons may be too many; two clear choices could reduce drop-off."})
    ideas.append({"name": "Card second", "target_segments": ["all"],
                  "config": {"order": ["paypal", "card", "venmo"], "paylater_msg": "below", "headline": "Choose a way to pay"},
                  "hypothesis": "Shoppers without PayPal should not have to scroll past three buttons to pay by card."})
    return ideas
