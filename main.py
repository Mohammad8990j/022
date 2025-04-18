import sys
from pathlib import Path
from datetime import datetime
from src.utils.logger import TradingBotLogger
from src.utils.config_loader import ConfigLoader

def setup_environment():
    """آماده‌سازی محیط اولیه برنامه"""
    # ایجاد پوشه‌های مورد نیاز
    directories = [
        "data/historical",
        "data/live",
        "data/processed",
        "models/cnn",
        "models/lstm",
        "models/transformer",
        "models/ppo",
        "models/dqn",
        "models/archived",
        "saved_models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory created/checked: {directory}")  # اضافه کردن print برای دیباگ

def main():
    # زمان شروع برنامه
    start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # تنظیم لاگر
    logger = TradingBotLogger("MainApp", log_level="DEBUG")  # تغییر سطح لاگ به DEBUG
    logger.info(f"=== شروع برنامه ربات تریدر در {start_time} ===")
    logger.info(f"کاربر فعلی: Mohammad8990j")
    
    try:
        # آماده‌سازی محیط
        logger.info("شروع آماده‌سازی محیط...")
        setup_environment()
        logger.info("محیط برنامه با موفقیت آماده شد")
        
        # بارگذاری تنظیمات
        logger.info("شروع بارگذاری تنظیمات...")
        config = ConfigLoader()
        config.load_all()
        logger.info("تنظیمات با موفقیت بارگذاری شدند")
        
        # نمایش برخی اطلاعات مهم از تنظیمات
        settings = config.settings
        if settings:
            logger.info(f"زبان سیستم: {settings.get('general', {}).get('language', 'نامشخص')}")
            logger.info(f"تم انتخاب شده: {settings.get('general', {}).get('theme', 'نامشخص')}")
        
        logger.info("=== برنامه با موفقیت راه‌اندازی شد ===")
        
    except Exception as e:
        logger.error(f"خطا در اجرای برنامه: {str(e)}")
        # اضافه کردن جزئیات بیشتر خطا
        import traceback
        logger.error(f"جزئیات خطا:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()