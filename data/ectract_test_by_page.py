import json
from pathlib import Path

# ===== PATHS =====
src_path = "/home/anhnd/gendata-websrc/stage1/data/raw_html/sportify_candidates.json"
out_path = "/home/anhnd/gendata-websrc/stage1/data/processed/sportify_queries.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_dataset(nodes):
    dataset = []

    # chỉ giữ candidate string
    candidates = [n.get("candidate", "") for n in nodes if n.get("candidate")]

    for i, node in enumerate(nodes):
        queries = node.get("queries", [])
        if not queries:
            continue

        positive = node.get("candidate", "")
        if not positive:
            continue

        # neg_candidates = tất cả candidate khác positive
        neg_candidates = [c for c in candidates if c != positive]

        for q in queries:
            dataset.append(
                {
                    "query": q,
                    "positive": positive,
                    "neg_candidates": neg_candidates,
                }
            )

    return dataset


def main():
    nodes = load_json(src_path)
    if not isinstance(nodes, list):
        raise ValueError("JSON input phải là list")

    dataset = build_dataset(nodes)
    save_json(dataset, out_path)

    print(f"✔ Generated {len(dataset)} samples")
    print(f"✔ Saved to {out_path}")


if __name__ == "__main__":
    main()
