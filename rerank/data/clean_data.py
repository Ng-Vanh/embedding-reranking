import json
# /mnt/disk2/anhnv/rr/stage1/data/processed_test/amazon_queries.json
# amazon_queries arxiv_queries awward_queries bbc_news_queries behance_queries economist_queries 
# health_queries neflix_queries pinterest_queries sportify_queries stackover_queries twitch_queries

text = "stackover_queries"
data_path = f"/mnt/disk2/anhnv/rr/stage1/data/processed_test/{text}.json"
data2_path = f"/mnt/disk2/anhnv/rr/stage2/data/stage2_{text}_top50.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(data2_path, "r", encoding="utf-8") as f:
    data2 = json.load(f)



list_texts_to_filter = [
"Enter q","Fill q"
]
new_data = []
print(f"Original data stage 1 samples: {len(data)}")
for item in data:
    query = item["query"]
    if any(text in query for text in list_texts_to_filter):
        continue
    new_data.append(item)
with open(data_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)
print(f"cleaned data stage 1 to: {data_path}, total samples: {len(new_data)}")

new_data2 = []
print(f"Original data stage 2 samples: {len(data2)}")
for item in data2:
    query = item["query"]
    if any(text in query for text in list_texts_to_filter):
        continue
    new_data2.append(item)
with open(data2_path, "w", encoding="utf-8") as f:
    json.dump(new_data2, f, indent=2, ensure_ascii=False)
print(f"cleaned data stage 2 to: {data2_path}, total samples: {len(new_data2)}")