import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from utils.logger import TradingBotLogger

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.logger = TradingBotLogger("FeatureEngineer")
        self.config = config
        self.processed_data_path = Path("data/processed")
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def process_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        add_indicators: bool = True,
        window_size: int = 20,
        save: bool = True
    ) -> pd.DataFrame:
        """
        پردازش داده‌های خام و اضافه کردن ویژگی‌ها
        
        Args:
            df: دیتافریم داده‌های خام
            symbol: نماد معاملاتی
            timeframe: تایم‌فریم
            add_indicators: اضافه کردن اندیکاتورها
            window_size: سایز پنجره برای ویژگی‌های زمانی
            save: ذخیره نتایج
        """
        try:
            # کپی از داده‌ها برای جلوگیری از تغییر داده‌های اصلی
            df = df.copy()
            
            # تبدیل ستون زمان به ایندکس
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
            
            # محاسبه تغییرات قیمت
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']).diff()
            
            # اضافه کردن اندیکاتورها
            if add_indicators:
                df = self.add_technical_indicators(df)
            
            # ایجاد ویژگی‌های زمانی
            df = self.add_temporal_features(df, window_size)
            
            # نرمال‌سازی داده‌ها
            df = self.normalize_features(df)
            
            # حذف ردیف‌های با مقادیر NaN
            df.dropna(inplace=True)
            
            if save:
                self._save_processed_data(df, symbol, timeframe)
            
            self.logger.info(
                f"پردازش داده‌های {symbol} انجام شد - "
                f"تعداد ویژگی‌ها: {len(df.columns)}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"خطا در پردازش داده‌ها: {str(e)}")
            raise
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """اضافه کردن اندیکاتورهای تکنیکال"""
        # RSI
        df['RSI'] = self._calculate_rsi(df['close'], period=14)
        
        # MACD
        df['MACD'], df['MACD_signal'] = self._calculate_macd(df['close'])
        
        # Moving Averages
        df['MA_20'] = df['close'].rolling(window=20).mean()
        df['MA_50'] = df['close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # MFI (Money Flow Index)
        df['MFI'] = self._calculate_mfi(df, period=14)
        
        # ATR (Average True Range)
        df['ATR'] = self._calculate_atr(df, period=14)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """محاسبه RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> tuple:
        """محاسبه MACD"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """محاسبه MFI"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        # محاسبه جریان مثبت و منفی
        price_diff = typical_price.diff()
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        # محاسبه نسبت جریان پول
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """محاسبه ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def add_temporal_features(
        self,
        df: pd.DataFrame,
        window_size: int = 20
    ) -> pd.DataFrame:
        """اضافه کردن ویژگی‌های زمانی"""
        # میانگین متحرک قیمت
        df['price_ma'] = df['close'].rolling(window=window_size).mean()
        
        # انحراف معیار قیمت
        df['price_std'] = df['close'].rolling(window=window_size).std()
        
        # تغییرات نسبی قیمت
        df['price_change'] = df['close'].pct_change(periods=window_size)
        
        # حجم معاملات نسبی
        df['volume_ma'] = df['volume'].rolling(window=window_size).mean()
        df['volume_std'] = df['volume'].rolling(window=window_size).std()
        
        # ویژگی‌های زمانی پیشرفته
        df['momentum'] = df['close'] - df['close'].shift(window_size)
        df['acceleration'] = df['momentum'].diff()
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """نرمال‌سازی ویژگی‌ها"""
        if exclude_cols is None:
            exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        # کپی از دیتافریم
        df_norm = df.copy()
        
        # نرمال‌سازی ستون‌های عددی
        for column in df_norm.columns:
            if column not in exclude_cols:
                series = df_norm[column].dropna()
                if len(series) > 0:
                    mean = series.mean()
                    std = series.std()
                    if std != 0:
                        df_norm[column] = (df_norm[column] - mean) / std
        
        return df_norm
    
    def _save_processed_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> None:
        """ذخیره داده‌های پردازش شده"""
        try:
            # اول سعی می‌کنیم با فرمت parquet ذخیره کنیم
            filename = (
                f"{symbol}_{timeframe}_processed_"
                f"{datetime.now():%Y%m%d_%H%M}.parquet"
            )
            try:
                df.to_parquet(self.processed_data_path / filename)
                self.logger.info(f"داده‌های پردازش شده در {filename} ذخیره شدند")
            except ImportError:
                # اگر pyarrow نصب نبود، از CSV استفاده می‌کنیم
                csv_filename = (
                    f"{symbol}_{timeframe}_processed_"
                    f"{datetime.now():%Y%m%d_%H%M}.csv"
                )
                df.to_csv(self.processed_data_path / csv_filename)
                self.logger.info(
                    f"داده‌های پردازش شده در {csv_filename} "
                    f"(با فرمت CSV) ذخیره شدند"
                )
        except Exception as e:
            self.logger.error(f"خطا در ذخیره داده‌ها: {str(e)}")
            raise