import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.preprocessing.feature_engineering import FeatureEngineer
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def sample_data():
    """ایجاد داده‌های نمونه برای تست"""
    dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='1H')
    df = pd.DataFrame({
        'time': dates,
        'open': np.random.uniform(1.1000, 1.2000, len(dates)),
        'high': np.random.uniform(1.1500, 1.2500, len(dates)),
        'low': np.random.uniform(1.0500, 1.1500, len(dates)),
        'close': np.random.uniform(1.1000, 1.2000, len(dates)),
        'volume': np.random.randint(100, 1000, len(dates))
    })
    return df

@pytest.fixture
def feature_engineer():
    """ایجاد نمونه FeatureEngineer"""
    config = ConfigLoader()
    config.load_all()
    return FeatureEngineer(config.settings)

def test_process_data(feature_engineer, sample_data):
    """تست پردازش کامل داده‌ها"""
    df_processed = feature_engineer.process_data(
        sample_data,
        symbol="XAUUSD_o",
        timeframe="H1",
        save=False
    )
    
    # بررسی وجود ویژگی‌های اصلی
    assert 'RSI' in df_processed.columns
    assert 'MACD' in df_processed.columns
    assert 'MFI' in df_processed.columns
    assert 'returns' in df_processed.columns
    
    # بررسی نرمال بودن داده‌ها
    for col in ['RSI', 'MACD', 'returns']:
        series = df_processed[col].dropna()
        assert -10 < series.mean() < 10
        assert 0 < series.std() < 10

def test_technical_indicators(feature_engineer, sample_data):
    """تست محاسبه اندیکاتورهای تکنیکال"""
    df = feature_engineer.add_technical_indicators(sample_data)
    
    # بررسی وجود همه اندیکاتورها
    assert 'RSI' in df.columns
    assert 'MACD' in df.columns
    assert 'MACD_signal' in df.columns
    assert 'MA_20' in df.columns
    assert 'MA_50' in df.columns
    assert 'BB_upper' in df.columns
    assert 'BB_lower' in df.columns
    assert 'MFI' in df.columns
    assert 'ATR' in df.columns

def test_temporal_features(feature_engineer, sample_data):
    """تست ایجاد ویژگی‌های زمانی"""
    df = feature_engineer.add_temporal_features(sample_data)
    
    # بررسی وجود ویژگی‌های زمانی
    assert 'price_ma' in df.columns
    assert 'price_std' in df.columns
    assert 'volume_ma' in df.columns
    assert 'momentum' in df.columns
    assert 'acceleration' in df.columns

def test_normalize_features(feature_engineer, sample_data):
    """تست نرمال‌سازی ویژگی‌ها"""
    # اضافه کردن یک ستون تست
    sample_data['test_column'] = np.random.normal(100, 15, len(sample_data))
    
    df_normalized = feature_engineer.normalize_features(
        sample_data,
        exclude_cols=['time']
    )
    
    # بررسی نرمال بودن ستون تست
    test_col = df_normalized['test_column'].dropna()
    assert -5 < test_col.mean() < 5
    assert 0.5 < test_col.std() < 1.5

def test_data_saving(feature_engineer, sample_data):
    """تست ذخیره‌سازی داده‌ها"""
    df_processed = feature_engineer.process_data(
        sample_data,
        symbol="EURUSD",
        timeframe="H1",
        save=True
    )
    
    # بررسی وجود فایل ذخیره شده (parquet یا CSV)
    from pathlib import Path
    processed_files_parquet = list(Path("data/processed").glob("EURUSD_H1_processed_*.parquet"))
    processed_files_csv = list(Path("data/processed").glob("EURUSD_H1_processed_*.csv"))
    
    # حداقل یکی از فرمت‌ها باید موجود باشد
    assert len(processed_files_parquet) > 0 or len(processed_files_csv) > 0