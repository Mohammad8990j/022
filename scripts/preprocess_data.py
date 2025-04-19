import pandas as pd
import talib
import os

# مسیر داده‌ها
RAW_DATA_DIR = "data/historical/"
PROCESSED_DATA_DIR = "data/processed/"
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

# افزودن اندیکاتورها
def add_indicators(df):
    """
    افزودن اندیکاتورهای مالی به داده‌ها.
    """
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MA'] = talib.SMA(df['close'], timeperiod=10)
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['tick_volume'], timeperiod=14)
    return df

# پیش‌پردازش داده‌ها
def preprocess_data(filename):
    """
    پیش‌پردازش یک فایل داده.
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath)

    # افزودن اندیکاتورها
    df = add_indicators(df)

    # حذف مقادیر NaN
    df.dropna(inplace=True)

    # ذخیره داده‌های پردازش‌شده
    save_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"داده‌های پردازش‌شده در مسیر {save_path} ذخیره شدند.")

def main():
    files = os.listdir(RAW_DATA_DIR)
    for file in files:
        if file.endswith(".csv"):
            preprocess_data(file)

if __name__ == "__main__":
    main()