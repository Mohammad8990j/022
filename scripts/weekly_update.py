import os
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
from scripts.preprocess_data import main as preprocess_data
from scripts.model_trainer import main as train_supervised_models
from src.models.reinforcement import train_ppo, train_dqn

# تنظیمات لاگینگ
LOG_FILE = "logs/weekly_update.log"
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def archive_old_models():
    """
    آرشیو کردن مدل‌های قدیمی.
    """
    try:
        MODEL_DIR = "saved_models/"
        ARCHIVE_DIR = "models/archived/"
        if not os.path.exists(ARCHIVE_DIR):
            os.makedirs(ARCHIVE_DIR)
        
        for file in os.listdir(MODEL_DIR):
            old_path = os.path.join(MODEL_DIR, file)
            new_path = os.path.join(ARCHIVE_DIR, file)
            os.rename(old_path, new_path)
            logging.info(f"مدل قدیمی {file} به آرشیو منتقل شد.")
    except Exception as e:
        logging.exception(f"خطا در آرشیو مدل‌ها: {e}")

def train_reinforcement_models():
    """
    آموزش مجدد مدل‌های یادگیری تقویتی.
    """
    try:
        logging.info("آموزش مدل PPO آغاز شد...")
        train_ppo(
            env_id="CartPole-v1",  # محیط نمونه
            timesteps=50000,
            save_path="saved_models/ppo_model.zip"
        )
        logging.info("آموزش مدل PPO به پایان رسید.")

        logging.info("آموزش مدل DQN آغاز شد...")
        train_dqn(
            env_id="CartPole-v1",  # محیط نمونه
            timesteps=50000,
            save_path="saved_models/dqn_model.zip"
        )
        logging.info("آموزش مدل DQN به پایان رسید.")
    except Exception as e:
        logging.exception(f"خطا در آموزش مدل‌های یادگیری تقویتی: {e}")

def main():
    try:
        logging.info("پیش‌پردازش داده‌ها آغاز شد...")
        preprocess_data()
        logging.info("پیش‌پردازش داده‌ها با موفقیت به پایان رسید.")

        logging.info("آرشیو مدل‌های قدیمی آغاز شد...")
        archive_old_models()
        logging.info("آرشیو مدل‌های قدیمی با موفقیت انجام شد.")

        logging.info("آموزش مجدد مدل‌های نظارتی آغاز شد...")
        train_supervised_models()
        logging.info("آموزش مدل‌های نظارتی با موفقیت به پایان رسید.")

        logging.info("آموزش مدل‌های یادگیری تقویتی آغاز شد...")
        train_reinforcement_models()
        logging.info("آموزش مدل‌های یادگیری تقویتی با موفقیت به پایان رسید.")

        logging.info("به‌روزرسانی هفتگی مدل‌ها با موفقیت انجام شد.")
    except Exception as e:
        logging.exception(f"خطا در به‌روزرسانی هفتگی: {e}")

if __name__ == "__main__":
    main()