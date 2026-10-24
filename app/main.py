from __future__ import annotations

import copy
import os
from pathlib import Path

_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():  # tiny .env loader, existing env vars win
    for line in _ENV.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm
from .bandit import LinTS
from .experiment import run
from .live import LiveLane
from .simulator import ACTION_LABELS

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Nudge Bandit")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

STATE: dict = {"sim": None, "model": None, "lane": None}


class SimParams(BaseModel):
    weeks: int = 26
    per_week: int = 20
    seed: int = 7


class Reply(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    lane = STATE["lane"]
    from .paypal import client
    return {"paypal_mode": lane.mode if lane else client().mode, "llm_provider": llm.provider() or "fallback",
            "action_labels": ACTION_LABELS}


@app.post("/api/simulate")
def simulate(p: SimParams):
    if not (4 <= p.weeks <= 52 and 5 <= p.per_week <= 100):
        raise HTTPException(400, "weeks must be 4-52 and per_week 5-100")
    r = run(weeks=p.weeks, per_week=p.per_week, seed=p.seed)
    STATE["model"] = LinTS.from_dict(r.pop("model"))
    STATE["sim"] = r
    return r


def _lane_view():
    lane: LiveLane = STATE["lane"]
    return {"mode": lane.mode, "epoch": lane.epoch, "invoices": lane.table(), "log": lane.log[-60:],
            "paypal_calls": len(lane.pp.calls)}


@app.post("/api/live/start")
def live_start():
    if STATE["model"] is None:
        simulate(SimParams())
    lane = LiveLane(copy.deepcopy(STATE["model"]))
    try:
        lane.start()
    except Exception as e:  # surface PayPal errors to the UI
        raise HTTPException(502, str(e))
    STATE["lane"] = lane
    return _lane_view()


@app.post("/api/live/step")
def live_step():
    if STATE["lane"] is None:
        raise HTTPException(400, "start the live lane first")
    try:
        STATE["lane"].step()
    except Exception as e:
        raise HTTPException(502, str(e))
    return _lane_view()


@app.post("/api/classify")
def classify(r: Reply):
    return llm.classify_reply(r.text)
