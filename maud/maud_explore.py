from datasets import load_dataset
from collections import Counter

ds = load_dataset('theatticusproject/maud')
all_data = ds['train']

# Unique text types (question categories)
text_types = set(all_data['text_type'])
print(f'Unique text_types: {len(text_types)}')
for t in sorted(text_types):
    print(f'  {t}')

# Unique answers
answers = set(all_data['answer'])
print(f'\nUnique answers: {len(answers)}')
print(sorted([a for a in answers if a is not None])[:30])

# Category breakdown
cats = Counter(all_data['category'])
print('\nCategories:')
for cat, count in cats.most_common():
    print(f'  {cat}: {count}')

# Total records across all splits
total = len(ds['train']) + len(ds['validation']) + len(ds['test'])
print(f'\nTotal train+val+test: {total}')
print(f'  train: {len(ds["train"])}')
print(f'  val  : {len(ds["validation"])}')
print(f'  test : {len(ds["test"])}')