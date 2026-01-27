import json

with open("/mnt/disk2/anhnv/rr/stage1/data/stage1_train_14_01.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Loaded {len(data)} entries from stage1_train_14_01.json")