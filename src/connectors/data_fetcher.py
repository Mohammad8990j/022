import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from mt5_connector import MT5Connector
from utils.logger import TradingBotLogger
from utils.error_handler import retry, safe_execute

class DataFetcher:
    def __init__(self, mt5_connector: MT5Connector):
        self.logger = TradingBotLogger("DataFetcher")
        self.mt5 = mt5_connector
        self.live_data_path = Path("data/live")
        self.live_data_path.mkdir(parents=True, exist_ok=True)

    @safe_execute
    def fetch_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        دریافت داده‌های تاریخی برای چند نماد
        
        Args:
            symbols: لیست نمادهای معاملاتی
            timeframe: تایم‌فریم
            start_date: تاریخ شروع
            end_date: تاریخ پایان (اختیاری)
        """
        data = {}
        for symbol in symbols:
            self.logger.info(f"دریافت داده‌های تاریخی برای {symbol}")
            data[symbol] = self.mt5.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                save=True
            )
        return data

    @retry(max_attempts=3, delay=1.0)
    def start_live_data_collection(
        self,
        symbols: List[str],
        interval: int = 1
    ) -> None:
        """
        شروع جمع‌آوری داده‌های لحظه‌ای
        
        Args:
            symbols: لیست نمادهای معاملاتی
            interval: فاصله زمانی بین هر درخواست (ثانیه)
        """
        import time
        
        self.logger.info(f"شروع جمع‌آوری داده‌های لحظه‌ای برای {symbols}")
        
        try:
            while True:
                for symbol in symbols:
                    # دریافت داده‌های لحظه‌ای
                    tick_data = self.mt5.get_realtime_data(symbol)
                    
                    # ذخیره در فایل
                    self._save_live_data(symbol, tick_data)
                    
                    self.logger.debug(
                        f"داده‌های {symbol} - "
                        f"Bid: {tick_data['bid']}, "
                        f"Ask: {tick_data['ask']}"
                    )
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("جمع‌آوری داده‌های لحظه‌ای متوقف شد")

    def _save_live_data(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> None:
        """ذخیره داده‌های لحظه‌ای"""
        filename = self.live_data_path / f"{symbol}_live_{datetime.now():%Y%m%d}.jsonl"
        
        # تبدیل datetime به string
        data['time'] = data['time'].strftime('%Y-%m-%d %H:%M:%S')
        
        # اضافه کردن به فایل
        with open(filename, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def load_latest_data(
        self,
        symbol: str,
        timeframe: str,
        n_candles: int = 1000
    ) -> pd.DataFrame:
        """
        بارگذاری آخرین داده‌های ذخیره شده
        
        Args:
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم
            n_candles: تعداد کندل‌های درخواستی
        """
        historical_path = Path("data/historical")
        files = list(historical_path.glob(f"{symbol}_{timeframe}_*.csv"))
        
        if not files:
            raise FileNotFoundError(f"داده‌ای برای {symbol} یافت نشد")
        
        # آخرین فایل را بخوان
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        df['time'] = pd.to_datetime(df['time'])
        
        return df.tail(n_candles)