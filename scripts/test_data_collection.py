
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from datetime import datetime, timedelta
from src.connectors.mt5_connector import MT5Connector
from src.connectors.data_fetcher import DataFetcher
from src.utils.config_loader import ConfigLoader

def main():
    # بارگذاری تنظیمات
    config = ConfigLoader()
    config.load_all()
    
    # ایجاد اتصال
    mt5 = MT5Connector(config.credentials)
    fetcher = DataFetcher(mt5)
    
    # تست دریافت داده‌های تاریخی
    symbols = ["EURUSD_o", "GBPUSD_o"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    historical_data = fetcher.fetch_historical_data(
        symbols=symbols,
        timeframe="M5",
        start_date=start_date,
        end_date=end_date
    )
    
    # نمایش اطلاعات داده‌های دریافت شده
    for symbol, data in historical_data.items():
        print(f"\nداده‌های {symbol}:")
        print(f"تعداد رکوردها: {len(data)}")
        print(f"بازه زمانی: از {data['time'].min()} تا {data['time'].max()}")
        print("\nنمونه داده:")
        print(data.head())
    
    # تست دریافت داده‌های لحظه‌ای
    print("\nشروع دریافت داده‌های لحظه‌ای (برای توقف Ctrl+C را فشار دهید)")
    fetcher.start_live_data_collection(symbols, interval=1)

if __name__ == "__main__":
    main()