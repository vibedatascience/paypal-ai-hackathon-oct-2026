"""PayPal Invoicing v2 client (sandbox) plus an in-memory mock.

Used calls:
  POST /v1/oauth2/token
  POST /v2/invoicing/invoices                 create draft invoice
  POST /v2/invoicing/invoices/{id}/send       send it to the client
  POST /v2/invoicing/invoices/{id}/remind     send a reminder with our subject + note
  POST /v2/invoicing/invoices/{id}/payments   record the payment when the client pays
  GET  /v2/invoicing/invoices/{id}            read status back from PayPal

Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET (sandbox app) to go live.
Without them the mock keeps the same interface so the app still runs.
"""
from __future__ import annotations

import datetime as dt
import itertools
import os
import threading
import time

import httpx

SANDBOX = "https://api-m.sandbox.paypal.com"


def _money(v: float) -> dict:
    return {"currency_code": "USD", "value": f"{v:.2f}"}


class PayPalSandbox:
    mode = "sandbox"

    def __init__(self, client_id: str, secret: str, base: str = SANDBOX):
        self.client_id, self.secret, self.base = client_id, secret, base
        self._token, self._exp = None, 0.0
        self._lock = threading.Lock()
        self.calls: list[dict] = []

    def _auth(self) -> str:
        with self._lock:  # one token fetch even when invoices are processed in parallel
            if self._token and time.time() < self._exp - 60:
                return self._token
            return self._fetch_token()

    def _fetch_token(self) -> str:
        r = httpx.post(f"{self.base}/v1/oauth2/token", auth=(self.client_id, self.secret),
                       data={"grant_type": "client_credentials"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        self._token, self._exp = j["access_token"], time.time() + j.get("expires_in", 3000)
        return self._token

    def _req(self, method: str, path: str, json: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self._auth()}", "Content-Type": "application/json",
                   "Prefer": "return=representation"}
        for attempt in range(3):  # retry transient sandbox errors
            r = httpx.request(method, f"{self.base}{path}", headers=headers, json=json, timeout=30)
            self.calls.append({"method": method, "path": path, "status": r.status_code})
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(1.5 * (attempt + 1))
        if r.status_code >= 400:
            raise RuntimeError(f"PayPal {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}

    def create_invoice(self, *, client_name: str, email: str, amount: float, po_number: str | None,
                       item: str, note: str) -> dict:
        today = dt.date.today().isoformat()
        body = {
            "detail": {"currency_code": "USD", "invoice_date": today, "note": note,
                       "payment_term": {"due_date": today}, **({"reference": po_number} if po_number else {})},
            "primary_recipients": [{"billing_info": {"name": {"given_name": client_name}, "email_address": email}}],
            "items": [{"name": item, "quantity": "1", "unit_amount": _money(amount)}],
        }
        return self._req("POST", "/v2/invoicing/invoices", body)

    def send_invoice(self, invoice_id: str) -> dict:
        return self._req("POST", f"/v2/invoicing/invoices/{invoice_id}/send",
                         {"send_to_recipient": True, "send_to_invoicer": False})

    def remind(self, invoice_id: str, subject: str, note: str) -> dict:
        return self._req("POST", f"/v2/invoicing/invoices/{invoice_id}/remind",
                         {"subject": subject, "note": note, "send_to_recipient": True, "send_to_invoicer": False})

    def record_payment(self, invoice_id: str, amount: float) -> dict:
        return self._req("POST", f"/v2/invoicing/invoices/{invoice_id}/payments",
                         {"method": "BANK_TRANSFER", "payment_date": dt.date.today().isoformat(), "amount": _money(amount),
                          "note": "Recorded by Smart Invoice Reminders demo"})

    def get_invoice(self, invoice_id: str) -> dict:
        return self._req("GET", f"/v2/invoicing/invoices/{invoice_id}")

    # Orders v2, used by the checkout page's PayPal buttons
    def create_order(self, amount: float, description: str = "Order") -> dict:
        return self._req("POST", "/v2/checkout/orders", {
            "intent": "CAPTURE",
            "purchase_units": [{"amount": {"currency_code": "USD", "value": f"{amount:.2f}"}, "description": description}],
        })

    def capture_order(self, order_id: str) -> dict:
        return self._req("POST", f"/v2/checkout/orders/{order_id}/capture", {})


class PayPalMock:
    """Same interface, no network. Status values mirror PayPal's."""
    mode = "mock"

    def __init__(self):
        self._seq = itertools.count(1)
        self.store: dict[str, dict] = {}
        self.calls: list[dict] = []

    def _log(self, method, path):
        self.calls.append({"method": method, "path": path, "status": 200})

    def create_invoice(self, *, client_name, email, amount, po_number, item, note):
        iid = f"INV2-MOCK-{next(self._seq):04d}"
        self.store[iid] = {"id": iid, "status": "DRAFT", "amount": amount, "reminders": []}
        self._log("POST", "/v2/invoicing/invoices")
        return {"id": iid, "status": "DRAFT"}

    def send_invoice(self, iid):
        self.store[iid]["status"] = "SENT"
        self._log("POST", f"/v2/invoicing/invoices/{iid}/send")
        return {}

    def remind(self, iid, subject, note):
        self.store[iid]["reminders"].append({"subject": subject, "note": note})
        self._log("POST", f"/v2/invoicing/invoices/{iid}/remind")
        return {}

    def record_payment(self, iid, amount):
        self.store[iid]["status"] = "MARKED_AS_PAID"
        self._log("POST", f"/v2/invoicing/invoices/{iid}/payments")
        return {}

    def get_invoice(self, iid):
        self._log("GET", f"/v2/invoicing/invoices/{iid}")
        return {"id": iid, "status": self.store[iid]["status"]}

    def create_order(self, amount, description="Order"):
        oid = f"MOCK-ORDER-{next(self._seq):04d}"
        self._log("POST", "/v2/checkout/orders")
        return {"id": oid, "status": "CREATED"}

    def capture_order(self, oid):
        self._log("POST", f"/v2/checkout/orders/{oid}/capture")
        return {"id": oid, "status": "COMPLETED"}


def client():
    cid, sec = os.getenv("PAYPAL_CLIENT_ID"), os.getenv("PAYPAL_CLIENT_SECRET")
    if cid and sec:
        return PayPalSandbox(cid, sec, os.getenv("PAYPAL_BASE", SANDBOX))
    return PayPalMock()
