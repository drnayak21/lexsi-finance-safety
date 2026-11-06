import json, random
from datasets import load_dataset

random.seed(0)
ds = load_dataset("gbharti/finance-alpaca", split="train")

KEYWORDS = ["stock", "bond", "invest", "interest rate", "loan", "mortgage", "credit",
            "tax", "inflation", "portfolio", "dividend", "equity", "mutual fund", "bank",
            "insurance", "retirement", "budget", "earnings", "revenue", "profit", "asset",
            "debt", "cash flow", "etf", "financial", "finance"]

def keep(r):
    out = (r.get("output") or "").strip()
    text = ((r["instruction"] or "") + " " + (r.get("input") or "") + " " + out).lower()
    return len(out.split()) >= 20 and any(k in text for k in KEYWORDS)

def to_messages(r):
    q = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": r["output"].strip()}]}

rows = [to_messages(r) for r in ds if keep(r)]
random.shuffle(rows)
print("kept", len(rows), "of", len(ds))
splits = {"train": rows[:1500], "eval": rows[1500:1700]}
for name, data in splits.items():
    with open(f"data/finance_{name}.jsonl", "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print(name, len(data))
for r in rows[:3]:
    print("-", r["messages"][0]["content"][:120].replace("\n", " "))
