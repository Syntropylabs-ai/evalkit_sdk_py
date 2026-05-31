import json
import os

import evalkit

evalkit.init(subscription_key=os.environ["EVALKIT_SUBSCRIPTION_KEY"])


def my_agent(ctx):
    return f"You said: {ctx.message}. How else can I help?"


scenarios = evalkit.generate_scenarios(
    "You are a customer-support agent for a SaaS billing product. "
    "Help users with charges, refunds, and subscription changes.",
    tools=["get_payment_history", "issue_refund", "change_plan"],
    count=3,
)

report = evalkit.simulate_user(my_agent, scenarios, tags=["example-run"])

print(json.dumps(report, indent=2))
