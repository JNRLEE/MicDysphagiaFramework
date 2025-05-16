# 這個腳本用於統計指定目錄中info和info.json檔案的數量，並將所有info檔案重命名為info.json
# # 先查看哪些檔案會被修改（不執行重命名）
# python rename_info_files.py /home/sbplab/JNRLEE/MicDysphagiaFramework/data/Processed\(Cut\) --dry-run

# # 執行實際重命名
# python rename_info_files.py /home/sbplab/JNRLEE/MicDysphagiaFramework/data/Processed\(Cut\)

import os
import argparse

def scan_and_count(dir_path):
    info_count = 0
    info_json_count = 0
    info_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('info') and not file.endswith('.info'):
                info_count += 1
                info_paths.append(os.path.join(root, file))
            elif file.endswith('info.json'):
                info_json_count += 1
    return info_count, info_json_count, info_paths

def rename_info(dir_path):
    renamed_count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('info') and not file.endswith('.info'):
                old_path = os.path.join(root, file)
                new_path = old_path + '.json'
                os.rename(old_path, new_path)
                print(f"已重命名: {old_path} -> {new_path}")
                renamed_count += 1
    return renamed_count

def main():
    parser = argparse.ArgumentParser(description="統計並重命名info檔案")
    parser.add_argument('path', nargs='?', default='data/Processed(Cut)', help="要掃描的目錄路徑")
    parser.add_argument('--dry-run', action='store_true', help="只顯示將被重命名的檔案，不執行重命名")
    args = parser.parse_args()
    
    info_count, info_json_count, info_paths = scan_and_count(args.path)
    print(f"以 info 結尾的檔案數量: {info_count}")
    print(f"以 info.json 結尾的檔案數量: {info_json_count}")
    
    if info_count > 0:
        print("\n以下為需要重命名的檔案:")
        for p in info_paths:
            print(f"{p} -> {p}.json")
        
        if not args.dry_run:
            renamed = rename_info(args.path)
            print(f"\n已成功重命名 {renamed} 個檔案")
        else:
            print("\n乾執行模式：未執行實際重命名")
    else:
        print("未發現需要重命名的檔案")

if __name__ == '__main__':
    main() 