import json

with open("/mnt/disk2/anhnv/rr/stage1/data/stage1_train_14_01_43562.json", "r",encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    query = item['query']
    positive = item['positive']