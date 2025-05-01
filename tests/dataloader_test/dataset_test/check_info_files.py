# 檢查所有子資料夾中的 info.json 檔案
# 並統計包含 score 的檔案數量

import os
import json
import glob

base_dir = "/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/WavData/Preparing"
count = 0
n_count = 0
zero_score = 0

for subdir in glob.glob(os.path.join(base_dir, "*/")):
    info_files = [f for f in glob.glob(os.path.join(subdir, "*info.json")) 
                 if not f.endswith("WavTokenizer_tokens_info.json")]
    
    if info_files:
        for info_file in info_files:
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "score" in data:
                        count += 1
                        if os.path.basename(os.path.dirname(info_file)).startswith("N"):
                            n_count += 1
                            print(f"Found N-prefix file: {info_file}, score: {data['score']}")
                        if data["score"] == 0:
                            zero_score += 1
                            print(f"Found score=0 in {info_file}")
            except Exception as e:
                print(f"Error reading {info_file}: {e}")

print(f"\nSummary:")
print(f"Total info.json files with score: {count}")
print(f"N-prefix files with score: {n_count}")
print(f"Files with score=0: {zero_score}") 