"""
配置加載模組：用於讀取和驗證YAML配置文件
功能：
1. 讀取YAML配置文件
2. 合併默認配置和用戶配置
3. 驗證配置合法性
4. 提供對配置的訪問方法
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加載器類，用於加載和解析YAML配置文件"""
    
    def __init__(self, config_path: Optional[str] = None, default_config_path: Optional[str] = None):
        """初始化配置加載器
        
        Args:
            config_path: YAML配置文件路徑
            default_config_path: 默認配置文件路徑（可選）
        """
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        self.default_config_path = default_config_path
        
        # 如果提供了配置路徑，立即加載
        if config_path:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加載配置文件
        
        Args:
            config_path: 配置文件路徑，如果提供則覆蓋初始化時的路徑
            
        Returns:
            Dict[str, Any]: 加載的配置字典
            
        Raises:
            FileNotFoundError: 如果找不到配置文件
            yaml.YAMLError: 如果YAML解析錯誤
        """
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            raise ValueError("未提供配置文件路徑")
        
        # 加載默認配置（如果有）
        default_config = {}
        if self.default_config_path and os.path.exists(self.default_config_path):
            try:
                with open(self.default_config_path, 'r', encoding='utf-8') as f:
                    default_config = yaml.safe_load(f)
                logger.info(f"已加載默認配置: {self.default_config_path}")
            except Exception as e:
                logger.warning(f"加載默認配置失敗: {str(e)}")
        
        # 加載用戶配置
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            logger.info(f"已加載用戶配置: {self.config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到配置文件: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML解析錯誤: {str(e)}")
        
        # 合併配置
        self._merge_configs(default_config, user_config)
        
        # 驗證配置
        self._validate_config()
        
        return self.config
    
    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
        """遞歸合併默認配置和用戶配置
        
        Args:
            default_config: 默認配置字典
            user_config: 用戶配置字典
        """
        # 如果默認配置為空，直接使用用戶配置
        if not default_config:
            self.config = user_config
            return
        
        # 遞歸合併
        self.config = self._recursive_merge(default_config, user_config)
    
    def _recursive_merge(self, default_dict: Dict[str, Any], user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """遞歸合併兩個字典，用戶配置優先
        
        Args:
            default_dict: 默認配置字典
            user_dict: 用戶配置字典
            
        Returns:
            Dict[str, Any]: 合併後的字典
        """
        result = default_dict.copy()
        
        for key, value in user_dict.items():
            # 如果用戶配置中的值是字典且默認配置中同名鍵也是字典，則遞歸合併
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._recursive_merge(result[key], value)
            else:
                # 否則用戶配置覆蓋默認配置
                result[key] = value
        
        return result
    
    def _validate_config(self) -> None:
        """驗證配置合法性，確保必要的配置項存在且有效"""
        # 檢查全局配置
        if 'global' not in self.config:
            logger.warning("配置中缺少 'global' 部分")
            self.config['global'] = {}
        
        # 檢查數據配置
        if 'data' not in self.config:
            raise ValueError("配置中缺少 'data' 部分")
        
        # 檢查模型配置
        if 'model' not in self.config:
            raise ValueError("配置中缺少 'model' 部分")
        
        # 檢查訓練配置
        if 'training' not in self.config:
            raise ValueError("配置中缺少 'training' 部分")
        
        # 檢查數據源
        data_type = self.config['data'].get('type')
        if data_type == 'audio' and not self.config['data'].get('source', {}).get('wav_dir'):
            logger.warning("音頻數據類型但未指定wav_dir")
        elif data_type == 'spectrogram' and not self.config['data'].get('source', {}).get('spectrogram_dir'):
            logger.warning("頻譜圖數據類型但未指定spectrogram_dir")
        elif data_type == 'feature' and not self.config['data'].get('source', {}).get('feature_dir'):
            logger.warning("特徵數據類型但未指定feature_dir")
    
    def get(self, key: str, default: Any = None) -> Any:
        """安全獲取配置項
        
        Args:
            key: 配置項鍵，可使用點號分隔的路徑，如 'data.batch_size'
            default: 如果鍵不存在時返回的默認值
            
        Returns:
            Any: 配置項的值，如果不存在則返回默認值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """通過字典式訪問配置項
        
        Args:
            key: 配置項鍵名
            
        Returns:
            Any: 配置項的值
            
        Raises:
            KeyError: 如果鍵不存在
        """
        return self.config[key]
    
    def save_config(self, save_path: Optional[str] = None) -> None:
        """保存當前配置到文件
        
        Args:
            save_path: 保存配置的文件路徑，如果不提供則使用原始配置路徑
            
        Raises:
            ValueError: 如果未提供保存路徑且原始配置路徑也不存在
        """
        if not save_path and not self.config_path:
            raise ValueError("未提供保存路徑")
        
        path = save_path or self.config_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"配置已保存到: {path}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            updates: 包含更新的配置項的字典
        """
        self._merge_configs(self.config, updates)


def load_config(config_path: str, default_config_path: Optional[str] = None) -> ConfigLoader:
    """便捷函數，加載配置並返回配置加載器實例
    
    Args:
        config_path: 配置文件路徑
        default_config_path: 默認配置文件路徑（可選）
        
    Returns:
        ConfigLoader: 配置加載器實例
    """
    loader = ConfigLoader(config_path, default_config_path)
    loader.load_config()
    return loader

# 中文註解：這是config_loader.py的Minimal Executable Unit，檢查ConfigLoader能正確加載、合併、驗證YAML配置，並測試錯誤情境時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for config_loader.py，檢查ConfigLoader能正確加載、合併、驗證YAML配置，並測試錯誤情境時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import os
    import yaml
    import logging
    logging.basicConfig(level=logging.INFO)
    # 建立臨時YAML檔案
    good_yaml = "test_good.yaml"
    bad_yaml = "test_bad.yaml"
    with open(good_yaml, "w") as f:
        yaml.dump({
            "global": {},
            "data": {"type": "audio", "source": {"wav_dir": "./"}},
            "model": {},
            "training": {}
        }, f)
    with open(bad_yaml, "w") as f:
        yaml.dump({"model": {}, "training": {}, "global": {}} , f)
    try:
        from utils.config_loader import ConfigLoader
        loader = ConfigLoader(good_yaml)
        loader.load_config()
        print("ConfigLoader測試成功")
    except Exception as e:
        print(f"ConfigLoader遇到錯誤（預期行為）: {e}")
    try:
        loader = ConfigLoader(bad_yaml)
        loader.load_config()
    except Exception as e:
        print(f"ConfigLoader遇到錯誤配置時的報錯（預期行為）: {e}")
    finally:
        os.remove(good_yaml)
        os.remove(bad_yaml) 