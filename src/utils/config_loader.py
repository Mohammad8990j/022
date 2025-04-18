import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.settings = {}
        self.strategies = {}
        self.credentials = {}
        
    def load_all(self) -> None:
        """بارگذاری تمام فایل‌های تنظیمات"""
        self.settings = self.load_yaml("settings.yaml")
        self.strategies = self.load_yaml("strategies.yaml")
        self.credentials = self.load_yaml("credentials.yaml")
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """بارگذاری یک فایل YAML"""
        try:
            with open(self.config_dir / filename, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"خطا در بارگذاری {filename}: {str(e)}")
    
    def save_yaml(self, data: Dict[str, Any], filename: str) -> None:
        """ذخیره تنظیمات در فایل YAML"""
        try:
            with open(self.config_dir / filename, 'w', encoding='utf-8') as file:
                yaml.dump(data, file, allow_unicode=True)
        except Exception as e:
            raise Exception(f"خطا در ذخیره {filename}: {str(e)}")