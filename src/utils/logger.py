import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

class TradingBotLogger:
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # اطمینان از اینکه logger قبلاً handler نداشته باشد
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # ایجاد پوشه logs اگر وجود ندارد
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # تنظیم فرمت لاگ با اطلاعات بیشتر
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # هندلر فایل
        current_date = datetime.now().strftime("%Y%m%d")
        file_handler = RotatingFileHandler(
            filename=log_dir / f"{name}_{current_date}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # هندلر کنسول با رنگ‌های مختلف برای سطوح مختلف لاگ
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # اضافه کردن هندلرها
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # لاگ تست برای اطمینان از کارکرد درست
        self.debug("Logger initialized successfully")
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)