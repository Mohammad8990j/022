import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import sys
from pathlib import Path
import pytest
from datetime import datetime, timedelta

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.connectors.mt5_connector import MT5Connector
from src.connectors.data_fetcher import DataFetcher
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def mt5_connector():
    """ایجاد یک اتصال MT5 برای تست"""
    config = ConfigLoader()
    config.load_all()
    return MT5Connector(config.credentials)

@pytest.fixture
def data_fetcher(mt5_connector):
    """ایجاد یک DataFetcher برای تست"""
    return DataFetcher(mt5_connector)

def test_mt5_connection(mt5_connector):
    """تست اتصال به MT5"""
    assert mt5_connector.check_connection()

def test_historical_data(mt5_connector):
    """تست دریافت داده‌های تاریخی"""
    # دریافت داده‌های یک هفته اخیر
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    df = mt5_connector.get_historical_data(
        symbol="XAUUSD_o",
        timeframe="M5",
        start_date=start_date,
        end_date=end_date
    )
    
    assert not df.empty
    assert 'time' in df.columns
    assert 'open' in df.columns
    assert 'close' in df.columns

def test_realtime_data(mt5_connector):
    """تست دریافت داده‌های لحظه‌ای"""
    tick_data = mt5_connector.get_realtime_data("EURUSD")
    
    assert 'bid' in tick_data
    assert 'ask' in tick_data
    assert tick_data['bid'] > 0
    assert tick_data['ask'] > 0

def test_data_fetcher(data_fetcher):
    """تست DataFetcher"""
    # دریافت داده‌های تاریخی برای چند نماد
    symbols = ["XAUUSD_o", "GBPUSD_o"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    data = data_fetcher.fetch_historical_data(
        symbols=symbols,
        timeframe="M5",
        start_date=start_date,
        end_date=end_date
    )
    
    assert len(data) == len(symbols)
    for symbol in symbols:
        assert symbol in data
        assert not data[symbol].empty