"""
測試音頻數據集的過濾邏輯
測試日期：2024-03-21
測試目的：驗證分類邏輯，特別是對於不在分數範圍內的資料處理
"""

import unittest
import os
import torch
from data.audio_dataset import AudioDataset
from utils.constants import LABEL_TO_INDEX, CLASS_LABELS

class TestAudioDataset(unittest.TestCase):
    def setUp(self):
        # 測試配置
        self.config = {
            'data': {
                'preprocessing': {
                    'audio': {
                        'sample_rate': 16000,
                        'duration': 5.0,
                        'normalize': True
                    }
                },
                'filtering': {
                    'score_thresholds': {
                        'normal': 0,  # score <= 0 為正常人
                        'patient': 9  # score >= 9 為病人
                    },
                    'class_config': {
                        'NoMovement': 1,
                        'DrySwallow': 1,
                        'Cracker': 1,
                        'Jelly': 1,
                        'WaterDrinking': 1
                    },
                    'subject_source': {
                        'normal': {
                            'include_N': 1,
                            'include_P': 1
                        },
                        'patient': {
                            'include_N': 0,
                            'include_P': 1
                        }
                    },
                    'task_type': 'classification'
                }
            }
        }

    def test_label_generation(self):
        """測試標籤生成邏輯"""
        print("\n=== 可用的類別標籤 ===")
        for label, idx in LABEL_TO_INDEX.items():
            print(f"{label}: {idx}")
            # 確保索引在0-9範圍內
            self.assertLess(idx, 10, f"標籤 {label} 的索引 {idx} 超出範圍")
            self.assertGreaterEqual(idx, 0, f"標籤 {label} 的索引 {idx} 小於0")

    def test_filtering_logic(self):
        """測試過濾邏輯，包括邊界條件和非正常/病人的資料"""
        test_dir = "test_audio_dataset"
        os.makedirs(test_dir, exist_ok=True)

        dataset = AudioDataset(test_dir, self.config)

        test_cases = [
            # 正常案例
            {
                'patient_id': 'N001',
                'score': 0,
                'selection': '無動作',
                'expected_group': 'Normal',
                'expected_type': 'NoMovement',
                'should_include': True
            },
            {
                'patient_id': 'P001',
                'score': 10,
                'selection': '乾吞嚥',
                'expected_group': 'Patient',
                'expected_type': 'DrySwallow',
                'should_include': True
            },
            # 邊界條件
            {
                'patient_id': 'N002',
                'score': -1,
                'selection': '無動作',
                'expected_group': 'Normal',
                'expected_type': 'NoMovement',
                'should_include': True
            },
            {
                'patient_id': 'P002',
                'score': 9,
                'selection': '果凍',
                'expected_group': 'Patient',
                'expected_type': 'Jelly',
                'should_include': True
            },
            # 不在範圍內的資料
            {
                'patient_id': 'P003',
                'score': 5,
                'selection': '餅乾',
                'expected_group': None,
                'expected_type': 'Cracker',
                'should_include': False
            },
            # 未定義的動作類型
            {
                'patient_id': 'P004',
                'score': 10,
                'selection': '未知動作',
                'expected_group': 'Patient',
                'expected_type': None,
                'should_include': False
            }
        ]

        print("\n=== 測試過濾邏輯 ===")
        for case in test_cases:
            print(f"\n測試案例:")
            print(f"  患者ID: {case['patient_id']}")
            print(f"  分數: {case['score']}")
            print(f"  選擇: {case['selection']}")
            print(f"  預期分組: {case['expected_group']}")
            print(f"  預期類型: {case['expected_type']}")
            
            # 檢查是否應該被包含在數據集中
            if case['should_include'] and case['expected_group'] and case['expected_type']:
                class_name = f"{case['expected_group']}-{case['expected_type']}"
                expected_label = LABEL_TO_INDEX.get(class_name)
                print(f"  預期標籤索引: {expected_label}")
                self.assertIsNotNone(expected_label, f"無效的類別名稱: {class_name}")
                self.assertLess(expected_label, 10, f"標籤索引 {expected_label} 超出範圍")
            else:
                print("  預期：此案例應被過濾掉")

    def test_dataset_filtering(self):
        """測試實際數據集過濾結果"""
        test_dir = "test_audio_dataset"
        os.makedirs(test_dir, exist_ok=True)

        dataset = AudioDataset(test_dir, self.config)
        
        # 檢查所有樣本的標籤
        for sample in dataset.samples:
            label = sample['label']
            self.assertIsInstance(label, torch.Tensor, "標籤應該是 torch.Tensor 類型")
            self.assertLess(label.item(), 10, f"標籤值 {label.item()} 超出範圍")
            self.assertGreaterEqual(label.item(), 0, f"標籤值 {label.item()} 小於0")

if __name__ == '__main__':
    unittest.main() 