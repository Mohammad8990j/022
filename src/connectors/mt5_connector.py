import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from utils.logger import TradingBotLogger
from utils.error_handler import retry, safe_execute

class MT5Connector:
    def __init__(self, config: Dict[str, Any]):
        self.logger = TradingBotLogger("MT5Connector")
        self.config = config
        self.connected = False
        self._initialize_mt5()

    @safe_execute
    def _initialize_mt5(self) -> None:
        """راه‌اندازی اولیه اتصال به MT5"""
        if not mt5.initialize(
            login=self.config['mt5']['login'],
            server=self.config['mt5']['server'],
            password=self.config['mt5']['password'],
            timeout=self.config['mt5']['timeout']
        ):
            raise ConnectionError(f"خطا در اتصال به MT5: {mt5.last_error()}")
        
        self.connected = True
        self.logger.info(f"اتصال به MT5 برقرار شد - نسخه: {mt5.__version__}")
        self._log_terminal_info()

    def _log_terminal_info(self) -> None:
        """ثبت اطلاعات ترمینال در لاگ"""
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            self.logger.info(
                f"نام کارگزار: {terminal_info.company}, "
                f"سرور: {terminal_info.path}, "
                f"حساب: {mt5.account_info().login}"
            )

    @retry(max_attempts=3, delay=1.0)
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        دریافت داده‌های تاریخی از MT5
        
        Args:
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم (مثال: 'M1', 'M5', 'H1', 'D1')
            start_date: تاریخ شروع
            end_date: تاریخ پایان (اختیاری)
            save: ذخیره داده‌ها در فایل
        """
        if not self.connected:
            raise ConnectionError("اتصال به MT5 برقرار نیست")

        # تبدیل تایم‌فریم به فرمت MT5
        timeframe_mt5 = getattr(mt5.TIMEFRAME_M1, timeframe, mt5.TIMEFRAME_M1)
        
        # دریافت داده‌ها
        rates = mt5.copy_rates_range(
            symbol,
            timeframe_mt5,
            start_date,
            end_date or datetime.now()
        )
        
        if rates is None:
            raise ValueError(f"خطا در دریافت داده‌های {symbol}: {mt5.last_error()}")
        
        # تبدیل به DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # ذخیره داده‌ها اگر درخواست شده باشد
        if save:
            self._save_historical_data(df, symbol, timeframe)
        
        self.logger.info(
            f"داده‌های تاریخی {symbol} دریافت شد - "
            f"تعداد: {len(df)}, از: {df['time'].min()}, تا: {df['time'].max()}"
        )
        
        return df

    def _save_historical_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> None:
        """ذخیره داده‌های تاریخی در فایل"""
        save_dir = Path("data/historical")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{symbol}_{timeframe}_{datetime.now():%Y%m%d}.csv"
        df.to_csv(save_dir / filename, index=False)
        self.logger.info(f"داده‌ها در {filename} ذخیره شدند")

    def get_realtime_data(self, symbol: str) -> Dict[str, float]:
        """دریافت داده‌های لحظه‌ای"""
        if not self.connected:
            raise ConnectionError("اتصال به MT5 برقرار نیست")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"خطا در دریافت تیک {symbol}: {mt5.last_error()}")
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time)
        }

    def check_connection(self) -> bool:
        """بررسی وضعیت اتصال"""
        return self.connected and mt5.terminal_info() is not None

    def __del__(self):
        """قطع اتصال هنگام حذف شیء"""
        if self.connected:
            mt5.shutdown()
            self.logger.info("اتصال MT5 قطع شد")