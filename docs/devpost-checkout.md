# Devpost submission text: Self-Optimizing Checkout for PayPal

**Tagline:** A PayPal checkout that improves itself: it watches where shoppers give up, Claude suggests a better layout, a statistical test proves it, and only winners are kept.

## In plain words

Every online store shows every shopper the same checkout. But a phone shopper, a big spender and a first-time buyer get stuck in different places. A phone shopper gives up at the PayPal password screen. A big cart feels expensive until it is shown as 4 smaller payments. A first-time buyer without a PayPal account has to scroll past three buttons to find the card option.

Self-Optimizing Checkout works like a store manager who never sleeps:

1. **Watch.** It logs every checkout: which button each shopper tapped and where they gave up.
2. **Suggest.** Claude reads those drop-offs and proposes a new layout, with a one-sentence reason a shop owner can understand.
3. **Test.** Some shoppers see the new layout. Layouts that more people pay with automatically get more shoppers.
4. **Keep.** Clear winners become the new default for that type of shopper. Bad ideas switch themselves off.

Then it repeats, separately for each type of shopper.

## What it does

The page shows the loop running. For any type of shopper (phone or desktop, small or big cart, new or returning) you see the original checkout and the checkout that shopper sees now, rendered with real PayPal JavaScript SDK buttons and Pay in 4 messages. Next to it is every idea Claude proposed, with the result of its test for each type of shopper: testing, kept, stopped, or beaten by a better idea.

In a run of 32,000 simulated shoppers, Claude proposed 9 layouts, 4 were kept, and checkout conversion rose from 46.1% to 49.5% on recent shoppers, about 340 more orders per 10,000 shoppers. Examples of what it learned on its own:

* New phone shoppers with small carts: a shorter list (PayPal, Venmo, card) lifted payment from 40.6% to 45.4%.
* New desktop shoppers with big carts: the Pay in 4 message above the buttons lifted payment from 47.8% to 52.1%.
* One of Claude's ideas ("Venmo up, Pay Later down") made things worse (38.0% vs 40.6%) and was switched off after 361 shoppers.

A goal setting lets the store optimize for all orders, or for checkouts paid with PayPal, Venmo or Pay Later.

## Why this is not an LLM wrapper

Claude never ships anything. It can only propose changes within three settings a merchant controls through the PayPal SDK: button order, where the Pay in 4 message appears, and the headline. Every proposal is validated, then has to win a statistical test before any shopper sees it by default.

## How we built it

* **Simulated shoppers** with hidden traits the system never sees (PayPal account, Venmo user, age), a choice model where earlier buttons get more attention, and step-by-step funnel logs.
* **Claude (Anthropic API)** reads the funnel per shopper segment and returns up to 3 experiments as structured JSON, with a plain-English hypothesis. A rule-based fallback keeps the loop running without a key.
* **Thompson sampling** with a Beta-Bernoulli model per shopper segment and layout. A layout is kept when it has at least a 95% chance of being the best; a test is stopped early when it is at least 97% likely to be worse than the current layout.
* **PayPal JavaScript SDK** renders the real PayPal, Venmo, Pay Later and card buttons and Pay in 4 messages in the preview, in the order each layout specifies.
* **PayPal Orders API v2 (sandbox)** creates and captures real orders when someone pays from the preview.
* **AG Grid and AG Charts** for the session log and charts; **FastAPI on Render** for hosting.

## Honest limits

The shoppers in the experiment are simulated, so the numbers show that the loop finds and keeps better layouts, not what a real store will see. The PayPal buttons, messages and orders are real sandbox calls.

## Links

* Live demo: https://paypal-ai-invoice-reminders.onrender.com/checkout
* Code: https://github.com/vibedatascience/paypal-ai-hackathon-oct-2026
