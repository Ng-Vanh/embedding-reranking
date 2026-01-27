import json

import os
import glob
FILE_PATTERN = "*.json"

folder_path = "/mnt/disk2/anhnv/rr/stage1/data/raw_html_train"
json_files = glob.glob(os.path.join(folder_path, FILE_PATTERN))
print(f"Found {len(json_files)} JSON files to merge.")
OUTPUT_FILE = "/mnt/disk2/anhnv/rr/stage1/data/stage1_train_extend_14_01.json"
import random
merged_data = []
for json_path in sorted(json_files):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for item in data:
                # Lấy danh sách queries, mặc định là danh sách trống nếu không có
                queries = item.get("queries")
                candidate = item.get("candidate")
                
                # Kiểm tra: 
                # 1. queries phải là list và không được trống (để tránh lỗi random.choice)
                # 2. candidate không được None (tùy nhu cầu của bạn)
                if isinstance(queries, list) and len(queries) > 0 and candidate is not None:
                    merged_data.append({
                        "query": "[ACTION] " + random.choice(queries),
                        "positive": candidate,
                    })
                else:
                    # Optional: log hoặc bỏ qua các trường hợp thiếu dữ liệu
                    continue
                    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {json_path}: {str(e)}")
    except Exception as e:
        print(f"Error reading file {json_path}: {str(e)}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
    json.dump(merged_data, out_f, ensure_ascii=False, indent=2)
print(f"Merged {len(merged_data)} JSON files into {OUTPUT_FILE}")
