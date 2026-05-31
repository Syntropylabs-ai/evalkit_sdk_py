import json
import os

import evalkit

evalkit.init(subscription_key=os.environ["EVALKIT_SUBSCRIPTION_KEY"])

scenarios = evalkit.generate_scenarios(
    "You are a customer-support agent for a SaaS billing product. "
    "Help users with charges, refunds, and subscription changes.",
    tools=["get_payment_history", "issue_refund", "change_plan"],
    count=3,
)

print(json.dumps(scenarios, indent=2))
