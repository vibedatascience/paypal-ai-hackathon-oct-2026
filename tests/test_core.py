from fastapi.testclient import TestClient

from app.experiment import run
from app.llm import classify_reply_fast, write_reminder
from app.main import app


def test_bandit_beats_baselines_across_seeds():
    for seed in (1, 2, 3):
        k = run(weeks=20, per_week=20, seed=seed)["kpis"]
        assert k["bandit"]["mean_days_to_paid"] < k["llm_wrapper"]["mean_days_to_paid"]
        assert k["bandit"]["mean_days_to_paid"] < k["llm_wrapper_frequent"]["mean_days_to_paid"]
        # Not winning by spamming: similar reminder volume to the wrapper.
        assert k["bandit"]["reminders_per_invoice"] < 1.5 * k["llm_wrapper"]["reminders_per_invoice"]


def test_fallback_classifier():
    assert classify_reply_fast("Our AP team needs a PO number on this") == "needs_po"
    assert classify_reply_fast("Sorry, totally forgot!") == "forgot"
    assert classify_reply_fast("Cash flow is tight this month") == "cash_tight"


def test_fallback_writer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    m = write_reminder("include_po", {"client_name": "Jordan", "amount": 4800, "days_overdue": 9, "po_number": "PO-1"})
    assert "PO-1" in m["note"] and m["subject"]


def test_live_lane_mock_end_to_end(monkeypatch):
    monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)  # always use the mock in tests
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = TestClient(app)
    assert c.post("/api/simulate", json={"weeks": 8, "per_week": 10}).status_code == 200
    v = c.post("/api/live/start").json()
    assert len(v["invoices"]) == 6 and all(i["paypal_status"] == "SENT" for i in v["invoices"])
    for _ in range(10):
        v = c.post("/api/live/step").json()
    assert any(i["paypal_status"] == "MARKED_AS_PAID" for i in v["invoices"])
    assert any(r["action"] != "Wait" for r in v["log"])


def test_checkout_loop_improves_and_validates(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.checkout import advisor, sim
    from app.checkout.engine import Optimizer
    o = Optimizer()
    o.run(2000, "baseline")
    for r in (1, 2):
        o.round = r
        advisor.propose(o)
        o.run(8000, "experiment")
        o.finalize()
    k = o.state()["kpi"]
    assert k["now"] > k["orig_now"] and k["kept"] >= 1
    assert not sim.valid_config({"order": ["venmo", "paylater"], "paylater_msg": "none", "headline": ""})[0]


def test_checkout_api_mock(monkeypatch):
    monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.checkout import api as capi
    capi.S["pp"] = None
    c = TestClient(app)
    assert c.post("/checkout/api/reset", json={"goal": "branded"}).json()["kpi"]["goal"] == "branded"
    c.post("/checkout/api/run", json={"sessions": 1000, "phase": "baseline"})
    s = c.post("/checkout/api/suggest").json()
    assert s["suggest"]["added"]
    oid = c.post("/checkout/api/orders", json={"amount": 64}).json()["id"]
    assert c.post(f"/checkout/api/orders/{oid}/capture").json()["status"] == "COMPLETED"
    assert c.get("/checkout").status_code == 200
