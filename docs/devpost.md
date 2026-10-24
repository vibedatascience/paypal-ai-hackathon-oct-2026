# Devpost submission text

**Project name:** Smart Invoice Reminders for PayPal

**Tagline:** An invoice reminder agent that learns which reminder gets each client to pay.

## Inspiration

Freelancers and small businesses send invoices through PayPal, and a large share get paid late. Every invoicing tool, and every LLM add-on, handles this the same way: a polite reminder on a fixed schedule. Clients pay late for different reasons. A corporate accounts payable team needs a PO number before its payment run. A startup founder forgot and will pay from a phone in one tap. A repeat late payer moves only when a late fee is mentioned. The same email to all three wastes weeks.

## What it does

The app treats each reminder as a decision and learns from outcomes. Every 3 days, for each unpaid PayPal invoice, it picks one of five actions: wait, a friendly email, a two-line note with a 1-tap PayPal link, a resend with the PO and billing details, or a firm note citing the contract late fee. Claude writes the reminder in that style and the app sends it through the PayPal Invoicing API `/remind` endpoint. When the client pays, the payment is recorded in PayPal and the model updates.

In a simulation of 520 invoices, identical for every policy, the learning agent got invoices paid in 12.8 days on average, compared with 25.3 days for an LLM wrapper that reads client replies but uses a fixed schedule, and 31.6 days for a plain fixed schedule. It sent the same number of reminders as the LLM wrapper (2.7 per invoice). Reminding every 3 days does not close the gap: that policy sent 8.6 reminders per invoice and still took 24.2 days.

## How we built it

* **Decision model:** disjoint linear Thompson sampling, one Bayesian linear regression per action over 10 context features (business or individual, late-payment history, invoice size, days overdue, reminders sent, whether an AP payment run is near, and the reason extracted from the client's reply). The reward is 1 if the invoice is paid before the next decision, minus a cost per reminder that grows after the second one, so the model learns to back off instead of spamming.
* **Claude (Anthropic API):** labels each client reply as needs_po, forgot, cash_tight or other, which becomes a model feature, and writes the reminder subject and note in the style the model chose.
* **PayPal Invoicing API v2 (sandbox):** create invoice, send, remind with the generated subject and note, record payment, and read status back.
* **AG Grid and AG Charts:** the learning curve, the learned expected reward per action, the live invoice table and the log of every reminder sent.
* **FastAPI + Render:** the app is deployed with a Render Blueprint.

## Challenges

Credit assignment for corporate clients: once the PO is on file, payment arrives at the next AP payment run, often after a later action. Making the payment run date a feature fixed this. Without a reminder cost, the model also learned to remind every epoch; a cost that grows after the second reminder fixed that while keeping the gain.

## What's next

The clients in the simulation are synthetic, so the numbers show that the method works, not what a real business will see. The next step is fitting the simulator to a real invoice history and running the model on live invoices.

## Built with

python, fastapi, numpy, claude, anthropic-api, paypal-invoicing-api, ag-grid, ag-charts, render

## Links

* Live demo: https://paypal-ai-invoice-reminders.onrender.com
* Code: https://github.com/vibedatascience/paypal-ai-hackathon-oct-2026
