from datasets import load_dataset
from collections import Counter

ds     = load_dataset("theatticusproject/maud")
test   = ds["test"]
counts = Counter(r["text_type"] for r in test if r.get("text_type"))

total = len(test)
print(f"Total test records: {total}")
print()
for qt, count in counts.most_common():
    pct = count / total * 100
    print(f"  {count:4d}  ({pct:5.1f}%)  {qt}")