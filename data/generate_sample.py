import pandas as pd
import random

samples = [
    ("I was charged twice for my subscription this month, please refund", "billing"),
    ("The app keeps crashing when I open settings", "technical_issue"),
    ("Can't login to my account, password reset email not arriving", "account"),
    ("I want a refund, the product didn't work as advertised", "refund"),
    ("Please add dark mode, it would be amazing!", "feature_request"),
    ("Your service is terrible, I've been waiting 3 days", "complaint"),
    ("Just wanted to say your team is incredible, thank you!", "praise"),
    ("Why is my invoice showing extra charges?", "billing"),
    ("Getting 500 error every time I upload a file", "technical_issue"),
    ("Love the new update, so much faster now", "praise"),
    ("Need to cancel my account immediately", "account"),
    ("The checkout page is broken on mobile", "technical_issue"),
    ("Please refund my last order, item never arrived", "refund"),
    ("Add export to PDF feature please", "feature_request"),
    ("Worst experience ever, canceling subscription", "complaint"),
]

rows = []
for i in range(200):
    text, _ = random.choice(samples)
    rows.append({"customer_id": f"CUST_{random.randint(1000,9999)}", "text": text})

pd.DataFrame(rows).to_csv("data/sample_tickets.csv", index=False)
print("✅ Generated 200 sample tickets")
