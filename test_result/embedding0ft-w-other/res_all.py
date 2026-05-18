import os
import json
import glob
from collections import defaultdict

 
# CONFIG
 
INPUT_DIR = "/mnt/disk2/anhnv/rr/stage1/test_result_v2/snowflake-arctic-embed-s"
OUTPUT_FILE = os.path.join(INPUT_DIR, "all_res.json")

# Pattern các file kết quả
PATTERN = os.path.join(INPUT_DIR, "test_results*.json")

 
# LOAD & AGGREGATE
 
files = glob.glob(PATTERN)

if not files:
    raise RuntimeError("Không tìm thấy file kết quả nào!")

sum_metrics = defaultdict(float)
count = 0

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chỉ lấy các metric dạng số (float / int)
    for k, v in data.items():
        if isinstance(v, (int, float)):
            sum_metrics[k] += v

    count += 1

 
# COMPUTE MEAN
 
avg_metrics = {
    "num_files": count
}

for k, v in sum_metrics.items():
    avg_metrics[f"avg_{k}"] = v / count

 
# SAVE RESULT
 
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(avg_metrics, f, indent=2, ensure_ascii=False)

print(f" Đã xử lý {count} file")
print(f" Kết quả trung bình lưu tại: {OUTPUT_FILE}")
