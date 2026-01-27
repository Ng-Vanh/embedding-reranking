# import json
# from tqdm import tqdm
# import re
# file = "/home/anhnd/gendata-websrc/rerank_1411/filtered_merged.json"
# output_file = "/home/anhnd/gendata-websrc/stage1/rr_stage1_data.json"
# with open(file, "r", encoding="utf-8") as f:
#     data = json.load(f)
# print(f"Loaded {len(data)} items from {file}")
# def extract_and_rebuild(s: str) -> str:
#     pattern = (
#         r"\[TAG\]\s*(?P<tag>\S+)\s*"
#         r"\[TID\]\s*\S+\s*"               # TID luôn bỏ
#         r"\[CLASS\]\s*(?P<class>.*?)\s*"  # CLASS có space
#         r"\[TEXT\]\s*(?P<text>.+)$"
#     )

#     m = re.search(pattern, s)
#     if not m:
#         return s  # không match thì giữ nguyên

#     tag = m.group("tag")
#     cls = m.group("class")
#     text = m.group("text")

#     # TID luôn bỏ
#     if cls == "NONE":
#         return f"[TAG] {tag} [TEXT] {text}"
#     else:
#         return f"[TAG] {tag} [CLASS] {cls} [TEXT] {text}"
# # print(extract_and_rebuild("[TAG] button [TID] 22 [CLASS] btn btn-default score_submitbutton [TEXT] Submit"))    

# new_data = []
# seen_query = set()
# for item in tqdm(data):
#     query = item["question"]
#     if query in seen_query:
#         continue
#     seen_query.add(query)
#     gold_idx = item["gold_idx"]
#     if gold_idx == len(item["nodes"]) - 1:
#         continue
#     positive_node = item["nodes"][gold_idx]


#     new_data.append({
#         "query": query,
#         "positive": extract_and_rebuild(positive_node),
#     })
# print(f"Converted to {len(new_data)} items for RR stage 1.")
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=4)

import json

f1 = "/home/anhnd/gendata-websrc/stage1/rr_stage1_data.json"
f2 = "/home/anhnd/gendata-websrc/stage1/stage1.json"
merg = "/home/anhnd/gendata-websrc/stage1/stage1_merged.json"

with open(f1, "r", encoding="utf-8") as file1, open(f2, "r", encoding="utf-8") as file2:
    data1 = json.load(file1)
    data2 = json.load(file2)    
with open(merg, "w", encoding="utf-8") as outfile:
    merged_data = data1 + data2
    print(f"Merged data has {len(merged_data)} items.") 
    json.dump(merged_data, outfile, ensure_ascii=False, indent=4)