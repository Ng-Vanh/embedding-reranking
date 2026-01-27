import json
import random
import re
from collections import defaultdict
from pathlib import Path


# CONFIG


INPUT_FILE = "/home/anhnd/gendata-websrc/stage1/data/stage1_merged.json"
TRAIN_OUT = "/home/anhnd/gendata-websrc/stage1/data/stage1_train_28160.json"
EVAL_OUT = "/home/anhnd/gendata-websrc/stage1/data/stage1_eval_2827_120neg.json"

NUM_EVAL = 2827
NUM_NEG = 120
SEED = 42

random.seed(SEED)


# HELPER FUNCTIONS


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

def jaccard(a, b):
    return len(a & b) / (len(a | b) + 1e-6)


# LOAD DATA


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) > NUM_EVAL, "Không đủ dữ liệu để tách eval!"

print(f"Total samples: {len(data)}")


# SHUFFLE & SPLIT


random.shuffle(data)

eval_raw = data[:NUM_EVAL]
train_data = data[NUM_EVAL:]

print(f"Eval size : {len(eval_raw)}")
print(f"Train size: {len(train_data)}")


# PREPARE NEGATIVE POOL


# Pool negatives = positive của train + eval (trừ chính nó khi dùng)
all_positives = [item["positive"] for item in data]

# Token cache cho nhanh
token_cache = {p: tokenize(p) for p in all_positives}


# BUILD EVAL WITH NEG CANDIDATES


eval_data = []

for idx, item in enumerate(eval_raw):
    query = item["query"]
    pos = item["positive"]

    pos_tokens = token_cache[pos]

    # Candidate negatives (loại chính positive)
    neg_pool = [p for p in all_positives if p != pos]

    # Tính lexical similarity
    scored = []
    for p in neg_pool:
        score = jaccard(pos_tokens, token_cache[p])
        scored.append((score, p))

    # Sort: hard negatives trước
    scored.sort(key=lambda x: x[0], reverse=True)

    # Lấy hard negatives + random fallback
    negs = []
    for _, p in scored:
        if p not in negs:
            negs.append(p)
        if len(negs) >= NUM_NEG:
            break

    # Nếu vẫn thiếu (hiếm)
    if len(negs) < NUM_NEG:
        remaining = list(set(neg_pool) - set(negs))
        negs.extend(random.sample(remaining, NUM_NEG - len(negs)))

    eval_data.append({
        "query": query,
        "positive": pos,
        "neg_candidates": negs
    })

    if (idx + 1) % 500 == 0:
        print(f"Built eval sample {idx + 1}/{len(eval_raw)}")


# SAVE FILES


Path(TRAIN_OUT).parent.mkdir(parents=True, exist_ok=True)

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(EVAL_OUT, "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print("DONE")
print(f"Train saved to: {TRAIN_OUT}")
print(f"Eval  saved to: {EVAL_OUT}")
