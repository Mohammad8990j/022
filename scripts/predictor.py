import os
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import torch
from src.models.cnn_model import CNNModel  # اصلاح ایمپورت
from src.models.lstm_model import LSTMModel  # اصلاح ایمپورت
from src.models.transformer_model import TransformerModel
from stable_baselines3 import PPO, DQN
import logging

# تنظیمات لاگینگ
LOG_FILE = "logs/predictor.log"
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# مسیر مدل‌ها
MODEL_DIR = "saved_models/"
PROCESSED_DATA_DIR = "data/processed/"
SIGNALS_DIR = "signals/"
if not os.path.exists(SIGNALS_DIR):
    os.makedirs(SIGNALS_DIR)

# بارگذاری مدل‌های نظارتی
def load_supervised_models():
    try:
        cnn_model = CNNModel(input_channels=3, output_size=3)
        cnn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "cnn_model.pt")))
        cnn_model.eval()

        lstm_model = LSTMModel(input_size=3, hidden_size=64, num_layers=2, output_size=1)
        lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lstm_model.pt")))
        lstm_model.eval()

        transformer_model = TransformerModel(input_size=20, num_heads=2, num_layers=2, output_size=3)
        transformer_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "transformer_model.pt")))
        transformer_model.eval()

        logging.info("مدل‌های نظارتی با موفقیت بارگذاری شدند.")
        return cnn_model, lstm_model, transformer_model
    except Exception as e:
        logging.exception(f"خطا در بارگذاری مدل‌های نظارتی: {e}")
        return None, None, None

# بارگذاری مدل‌های تقویتی
def load_reinforcement_models():
    try:
        ppo_model = PPO.load(os.path.join(MODEL_DIR, "ppo_model.zip"))
        dqn_model = DQN.load(os.path.join(MODEL_DIR, "dqn_model.zip"))

        logging.info("مدل‌های تقویتی با موفقیت بارگذاری شدند.")
        return ppo_model, dqn_model
    except Exception as e:
        logging.exception(f"خطا در بارگذاری مدل‌های تقویتی: {e}")
        return None, None

# پیش‌بینی با مدل‌های نظارتی
def predict_with_supervised_models(cnn_model, lstm_model, transformer_model, data):
    try:
        # تبدیل داده‌ها به نوع عددی
        data = data.astype(float)
        
        # اطمینان از اینکه داده‌ها دارای ابعاد مناسب هستند
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"داده‌های ورودی دارای ابعاد نادرست هستند: {data.shape}. انتظار [Batch, Features >= 3].")

        # تبدیل داده‌ها به قالب [Batch Size, Channels, Features]
        data_cnn = torch.tensor(data).float()
        
        # اطمینان از داشتن 3 کانال (تکرار داده‌ها در صورت نیاز)
        if data_cnn.shape[1] == 1:
            data_cnn = data_cnn.repeat(1, 3, 1)  # تکرار کانال‌ها برای ایجاد 3 کانال
        
        data_cnn = data_cnn.unsqueeze(1)  # تبدیل به [Batch Size, Channels, Length]

        # پیش‌بینی با مدل‌ها
        cnn_signals = cnn_model(data_cnn).argmax(dim=1).tolist()
        lstm_signals = lstm_model(torch.tensor(data).float()).detach().squeeze().tolist()
        transformer_signals = transformer_model(torch.tensor(data).float()).argmax(dim=1).tolist()

        logging.info("پیش‌بینی با مدل‌های نظارتی انجام شد.")
        return cnn_signals, lstm_signals, transformer_signals
    except Exception as e:
        logging.exception(f"خطا در پیش‌بینی با مدل‌های نظارتی: {e}")
        return None, None, None
# تولید سیگنال‌های معاملاتی
def generate_signals():
    try:
        # بارگذاری مدل‌ها
        cnn_model, lstm_model, transformer_model = load_supervised_models()
        ppo_model, dqn_model = load_reinforcement_models()

        if not all([cnn_model, lstm_model, transformer_model, ppo_model, dqn_model]):
            logging.error("تمام مدل‌ها بارگذاری نشدند. فرآیند متوقف شد.")
            return

        # بارگذاری داده‌های جدید
        data_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".parquet")]
        latest_file = sorted(data_files)[-1]
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, latest_file)).dropna().values

        # بررسی و تبدیل داده‌ها به نوع عددی
        try:
            data = data.astype(float)
        except ValueError as e:
            logging.error(f"خطا در تبدیل داده‌ها به نوع عددی: {e}")
            return

        # پیش‌بینی با مدل‌های نظارتی
        cnn_signals, lstm_signals, transformer_signals = predict_with_supervised_models(
            cnn_model, lstm_model, transformer_model, data[:, :-1]
        )

        # ذخیره سیگنال‌ها
        signals_df = pd.DataFrame({
            "time": pd.to_datetime(data[:, 0], unit='s'),
            "cnn_signal": cnn_signals,
            "lstm_signal": lstm_signals,
            "transformer_signal": transformer_signals,
        })
        signals_file = os.path.join(SIGNALS_DIR, "signals.csv")
        signals_df.to_csv(signals_file, index=False)
        logging.info(f"سیگنال‌های معاملاتی در مسیر {signals_file} ذخیره شدند.")

    except Exception as e:
        logging.exception(f"خطا در فرآیند تولید سیگنال‌ها: {e}")

def main():
    logging.info("فرآیند تولید سیگنال‌های معاملاتی آغاز شد.")
    generate_signals()
    logging.info("فرآیند تولید سیگنال‌های معاملاتی با موفقیت به پایان رسید.")

if __name__ == "__main__":
    main()