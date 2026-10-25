"""HTTP API for the Self-Optimizing Checkout page."""
from __future__ import annotations

import os
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import llm, paypal
from . import advisor, sim
from .engine import Optimizer

router = APIRouter()
LOCK = threading.Lock()
S = {"opt": Optimizer(), "pp": None, "real_orders": []}


def pp():
    if S["pp"] is None:
        S["pp"] = paypal.client()
    return S["pp"]


class RunReq(BaseModel):
    sessions: int = 1000
    phase: str = "experiment"


class ResetReq(BaseModel):
    goal: str = "orders"


class OrderReq(BaseModel):
    amount: float
    segment: str = ""
    layout: str = ""


@router.get("/api/config")
def config():
    return {"client_id": os.getenv("PAYPAL_CLIENT_ID", ""), "paypal_mode": pp().mode,
            "llm": llm.provider() or "rules", "segments": [{"seg": s, "label": sim.seg_label(s)} for s in sim.SEGMENTS],
            "original": sim.ORIGINAL, "methods": sim.METHOD_LABEL}


@router.get("/api/state")
def state():
    with LOCK:
        st = S["opt"].state()
    st["real_orders"] = S["real_orders"][-20:][::-1]
    return st


@router.post("/api/reset")
def reset(r: ResetReq | None = None):
    goal = r.goal if r and r.goal in ("orders", "branded") else "orders"
    with LOCK:
        S["opt"] = Optimizer(goal=goal)
    return state()


@router.post("/api/run")
def run(r: RunReq):
    if not 100 <= r.sessions <= 20000:
        raise HTTPException(400, "sessions must be 100-20000")
    with LOCK:
        S["opt"].run(r.sessions, r.phase)
    return state()


@router.post("/api/suggest")
def suggest():
    with LOCK:
        opt = S["opt"]
        if not opt.funnels:
            raise HTTPException(400, "watch some shoppers first")
        opt.round += 1
        res = advisor.propose(opt)
    out = state()
    out["suggest"] = res
    return out


@router.post("/api/decide")
def decide():
    with LOCK:
        S["opt"].finalize()
    return state()


@router.post("/api/orders")
def create_order(o: OrderReq):
    try:
        res = pp().create_order(o.amount, f"Checkout test · {o.layout or 'layout'}")
    except Exception as e:
        raise HTTPException(502, str(e))
    S["real_orders"].append({"order_id": res["id"], "status": res.get("status", ""), "amount": o.amount,
                             "segment": o.segment, "layout": o.layout})
    return {"id": res["id"]}


@router.post("/api/orders/{oid}/capture")
def capture(oid: str):
    try:
        res = pp().capture_order(oid)
    except Exception as e:
        raise HTTPException(502, str(e))
    for r in S["real_orders"]:
        if r["order_id"] == oid:
            r["status"] = res.get("status", "COMPLETED")
    return {"id": oid, "status": res.get("status")}
