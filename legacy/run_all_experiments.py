"""
實驗批次執行模組：運行多個配置的實驗
功能：
1. 依次執行多個實驗配置
2. 收集並比較實驗結果
3. 生成總結報告
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    """解析命令行參數
    
    Returns:
        argparse.Namespace: 解析後的參數
        
    Description:
        解析命令行參數，設置實驗配置
        
    References:
        https://docs.python.org/3/library/argparse.html
    """
    parser = argparse.ArgumentParser(description='批次運行多個吞嚥障礙評估模型實驗')
    
    parser.add_argument('--configs', nargs='+', type=str, default=None, 
                        help='要運行的配置文件列表')
    parser.add_argument('--output_dir', type=str, default='results/batch_experiments',
                        help='批次實驗結果輸出目錄')
    parser.add_argument('--device', type=str, default='auto',
                        help='使用的設備，例如cuda:0或cpu')
    parser.add_argument('--epochs', type=int, default=None,
                        help='每個實驗的訓練輪數，覆蓋配置文件中的值')
    parser.add_argument('--log_only', action='store_true',
                        help='僅記錄實驗結果，不訓練模型')
    
    return parser.parse_args()

def get_configs():
    """獲取配置文件列表
    
    Returns:
        list: 配置文件路徑列表
        
    Description:
        獲取config目錄下的所有YAML配置文件
        
    References:
        None
    """
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    configs = [
        os.path.join(config_dir, f) for f in os.listdir(config_dir)
        if f.endswith('.yaml') and not f == 'config_schema.yaml'
    ]
    return configs

def run_experiment(config_path, output_dir, device=None, epochs=None):
    """運行單個實驗
    
    Args:
        config_path: 配置文件路徑
        output_dir: 輸出目錄
        device: 使用的設備
        epochs: 覆蓋配置文件中的訓練輪數
        
    Returns:
        dict: 實驗結果
        
    Description:
        運行單個實驗配置，記錄結果
        
    References:
        None
    """
    print(f"\n{'='*80}")
    print(f"運行實驗: {config_path}")
    print(f"{'='*80}")
    
    # 構建命令
    cmd = ['python', 'scripts/run_experiments.py', '--config', config_path]
    
    if output_dir:
        cmd.extend(['--output_dir', output_dir])
    
    if device:
        cmd.extend(['--device', device])
    
    start_time = time.time()
    
    try:
        # 運行實驗
        process = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        # 解析輸出
        output = process.stdout
        error = process.stderr
        
        if error:
            print(f"警告: {error}")
        
        # 從輸出中提取實驗ID
        experiment_id = None
        for line in output.split('\n'):
            if '實驗開始：' in line:
                experiment_id = line.split('實驗開始：')[1].strip()
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'config_path': config_path,
            'success': True,
            'experiment_id': experiment_id,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"實驗完成: {config_path}")
        print(f"耗時: {duration:.2f} 秒")
        return result
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'config_path': config_path,
            'success': False,
            'error': str(e),
            'returncode': e.returncode,
            'output': e.stdout,
            'stderr': e.stderr,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"實驗失敗: {config_path}")
        print(f"錯誤: {e}")
        return result

def save_batch_results(results, output_dir):
    """保存批次實驗結果
    
    Args:
        results: 實驗結果列表
        output_dir: 輸出目錄
        
    Description:
        將批次實驗結果保存到JSON文件
        
    References:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(output_dir, f'batch_results_{timestamp}.json')
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n批次實驗結果已保存到: {result_file}")

def print_summary(results):
    """打印實驗結果摘要
    
    Args:
        results: 實驗結果列表
        
    Description:
        打印批次實驗的摘要統計
        
    References:
        None
    """
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    total_time = sum(r['duration'] for r in results)
    
    print("\n" + "="*80)
    print(f"批次實驗摘要:")
    print(f"總實驗數: {total_count}")
    print(f"成功實驗數: {success_count}")
    print(f"失敗實驗數: {total_count - success_count}")
    print(f"總耗時: {total_time:.2f} 秒 ({total_time/60:.2f} 分鐘)")
    
    if success_count > 0:
        print("\n成功實驗詳情:")
        for r in results:
            if r['success']:
                config_name = os.path.basename(r['config_path'])
                print(f"- {config_name}: 實驗ID {r.get('experiment_id', 'unknown')}, 耗時 {r['duration']:.2f} 秒")
    
    if success_count < total_count:
        print("\n失敗實驗詳情:")
        for r in results:
            if not r['success']:
                config_name = os.path.basename(r['config_path'])
                print(f"- {config_name}: 錯誤碼 {r.get('returncode', 'unknown')}")
                if 'error' in r:
                    print(f"  錯誤信息: {r['error']}")

def main():
    """主函數
    
    Description:
        批次實驗執行的主入口
        
    References:
        None
    """
    args = parse_args()
    
    # 獲取配置文件列表
    if args.configs:
        configs = args.configs
    else:
        configs = get_configs()
    
    if not configs:
        print("錯誤: 沒有找到配置文件")
        return
    
    print(f"找到 {len(configs)} 個配置文件:")
    for config in configs:
        print(f"- {config}")
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 運行模擬或實際實驗
    results = []
    
    if args.log_only:
        print("\n僅記錄模式，不執行實際訓練")
        for config in configs:
            results.append({
                'config_path': config,
                'success': True,
                'experiment_id': f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'duration': 0,
                'timestamp': datetime.now().isoformat(),
                'simulation': True
            })
    else:
        for config in configs:
            result = run_experiment(
                config_path=config,
                output_dir=args.output_dir,
                device=args.device,
                epochs=args.epochs
            )
            results.append(result)
    
    # 保存批次結果
    save_batch_results(results, args.output_dir)
    
    # 打印摘要
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc() 