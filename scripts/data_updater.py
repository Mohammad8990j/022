import MetaTrader5 as mt5
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# تنظیمات لاگینگ
LOG_FILE = "logs/data_updater.log"
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# مسیر ذخیره‌سازی داده‌ها
DATA_DIR = "data/historical/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# اتصال به MT5
def connect_to_mt5():
    if not mt5.initialize():
        logging.error(f"خطا در اتصال به MT5: {mt5.last_error()}")
        return False
    logging.info("اتصال به MT5 برقرار شد.")
    return True

# دریافت داده‌های جدید
def fetch_new_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, days=7):
    """
    دریافت داده‌های جدید از MT5.
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        if rates is None:
            logging.error(f"خطا در دریافت داده‌ها برای {symbol}: {mt5.last_error()}")
            return None

        # تبدیل به DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        logging.info(f"داده‌های جدید برای {symbol} دریافت شدند.")
        return df
    except Exception as e:
        logging.exception(f"خطایی در دریافت داده‌های {symbol}: {e}")
        return None

# ذخیره داده‌ها
def save_data(df, symbol):
    try:
        filename = os.path.join(DATA_DIR, f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}.csv")
        df.to_csv(filename, index=False)
        logging.info(f"داده‌های جدید در مسیر {filename} ذخیره شدند.")
    except Exception as e:
        logging.exception(f"خطا در ذخیره داده‌های {symbol}: {e}")

def main():
    if not connect_to_mt5():
        return

    # دریافت داده‌ها برای EURUSD
    symbol = "EURUSD"
    df = fetch_new_data(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, days=7)
    if df is not None:
        save_data(df, symbol)

    mt5.shutdown()
    logging.info("سیستم MT5 خاموش شد.")

if __name__ == "__main__":
    main()