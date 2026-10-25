"""Simulated shoppers on a merchant checkout page that uses PayPal's buttons.

Every shopper belongs to a visible segment (device, cart size, new or returning)
and has hidden traits the system never sees: whether they have a PayPal account,
whether they use Venmo, and whether they are under 35.

A checkout "UX config" is what the merchant controls through the PayPal
JavaScript SDK:
  order        which payment buttons appear and in what order
               (paypal, venmo, paylater, card)
  paylater_msg where the "Pay in 4" message appears: none, above, below
  headline     the line of text above the buttons

The shopper looks at the buttons (earlier buttons get more attention), picks one
or leaves, then either completes that payment method or drops off. Each step is
written to the session log, which is what the optimizer reads.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

METHODS = ["paypal", "venmo", "paylater", "card"]
BRANDED = {"paypal", "venmo", "paylater"}
METHOD_LABEL = {"paypal": "PayPal", "venmo": "Venmo", "paylater": "Pay Later", "card": "Debit or credit card"}
STEP_LABEL = {"paypal": "PayPal login", "venmo": "Venmo approval", "paylater": "Pay Later approval", "card": "Card form"}
POS_WEIGHT = [1.0, 0.68, 0.48, 0.36]

DEVICES = ["mobile", "desktop"]
CARTS = ["small", "big"]
SHOPPERS = ["new", "returning"]
SEGMENTS = [f"{d}|{c}|{s}" for d in DEVICES for c in CARTS for s in SHOPPERS]


def seg_label(seg: str) -> str:
    d, c, s = seg.split("|")
    return f"{d.title()} · {'Big' if c == 'big' else 'Small'} cart · {'New' if s == 'new' else 'Returning'} shopper"


ORIGINAL = {"order": ["paypal", "paylater", "venmo", "card"], "paylater_msg": "none", "headline": "Choose a way to pay"}


@dataclass
class Shopper:
    sid: int
    device: str
    cart: str
    shopper: str
    amount: float
    young: bool
    has_venmo: bool
    has_paypal: bool

    @property
    def segment(self) -> str:
        return f"{self.device}|{self.cart}|{self.shopper}"


def make_shopper(sid: int, rng: random.Random) -> Shopper:
    device = "mobile" if rng.random() < 0.6 else "desktop"
    cart = "big" if rng.random() < 0.3 else "small"
    shopper = "returning" if rng.random() < 0.45 else "new"
    amount = round(rng.uniform(210, 900) if cart == "big" else rng.uniform(18, 140), 2)
    young = rng.random() < (0.55 if device == "mobile" else 0.35)
    has_venmo = rng.random() < (0.72 if young else 0.15)
    has_paypal = rng.random() < (0.86 if shopper == "returning" else 0.33)
    return Shopper(sid, device, cart, shopper, amount, young, has_venmo, has_paypal)


def _appeal(m: str, sh: Shopper, cfg: dict) -> float:
    if m == "paypal":
        return 1.0 if sh.has_paypal else 0.15
    if m == "venmo":
        if not sh.has_venmo:
            return 0.03
        return 1.25 if sh.device == "mobile" else 0.45
    if m == "paylater":
        if sh.cart == "big":
            return 0.3 + {"above": 0.6, "below": 0.25, "none": 0.0}[cfg["paylater_msg"]]
        return 0.05 + (0.02 if cfg["paylater_msg"] != "none" else 0.0)
    return 0.35 if sh.has_paypal else 0.95  # card


def _complete(m: str, sh: Shopper) -> float:
    if m == "paypal":
        if not sh.has_paypal:
            return 0.35  # has to create an account
        return 0.70 if sh.device == "mobile" else 0.88  # typing a password on a phone
    if m == "venmo":
        return 0.93 if sh.device == "mobile" else 0.52  # desktop needs a QR scan
    if m == "paylater":
        return 0.78
    return 0.60 if sh.device == "mobile" else 0.80  # typing a card number


def _leave(sh: Shopper, cfg: dict) -> float:
    base = 0.62 if sh.device == "mobile" else 0.48
    return base + 0.06 * max(0, len(cfg["order"]) - 3)  # clutter


def choice_probs(sh: Shopper, cfg: dict) -> dict:
    w = {m: _appeal(m, sh, cfg) * POS_WEIGHT[i] for i, m in enumerate(cfg["order"])}
    tot = sum(w.values()) + _leave(sh, cfg)
    out = {m: v / tot for m, v in w.items()}
    out["leave"] = _leave(sh, cfg) / tot
    return out


def expected_conversion(sh: Shopper, cfg: dict) -> tuple[float, float]:
    """(probability of paying, probability of paying with a PayPal-branded method)."""
    p = choice_probs(sh, cfg)
    conv = sum(p[m] * _complete(m, sh) for m in cfg["order"])
    branded = sum(p[m] * _complete(m, sh) for m in cfg["order"] if m in BRANDED)
    return conv, branded


def run_session(sh: Shopper, cfg: dict, rng: random.Random) -> dict:
    """Simulate one checkout. Returns the outcome and the event log."""
    p = choice_probs(sh, cfg)
    events = [("checkout_view", ",".join(cfg["order"]))]
    r, acc, pick = rng.random(), 0.0, "leave"
    for m in cfg["order"] + ["leave"]:
        acc += p[m]
        if r < acc:
            pick = m
            break
    if pick == "leave":
        events.append(("abandon", "buttons"))
        return {"paid": False, "method": None, "drop": "buttons", "events": events}
    pos = cfg["order"].index(pick)
    events.append(("button_click", f"{pick}@{pos + 1}"))
    events.append(("auth_step", pick))
    if rng.random() < _complete(pick, sh):
        events.append(("paid", pick))
        return {"paid": True, "method": pick, "drop": None, "events": events}
    events.append(("abandon", pick))
    return {"paid": False, "method": pick, "drop": pick, "events": events}


def valid_config(cfg: dict) -> tuple[bool, str]:
    order = cfg.get("order")
    if not isinstance(order, list) or not 2 <= len(order) <= 4:
        return False, "order needs 2 to 4 payment options"
    if len(set(order)) != len(order) or any(m not in METHODS for m in order):
        return False, "order has unknown or repeated options"
    if "paypal" not in order or "card" not in order:
        return False, "PayPal and card must always be offered"
    if cfg.get("paylater_msg") not in ("none", "above", "below"):
        return False, "paylater_msg must be none, above or below"
    if not isinstance(cfg.get("headline", ""), str) or len(cfg.get("headline", "")) > 48:
        return False, "headline must be 48 characters or fewer"
    return True, ""
