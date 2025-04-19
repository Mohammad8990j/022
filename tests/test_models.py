import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn_model import CNNModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel


@pytest.fixture
def sample_data():
    """ایجاد داده‌های نمونه برای تست"""
    # داده‌های تصادفی برای آزمایش
    inputs = torch.rand(100, 3, 50)  # 100 نمونه، 3 کانال ورودی، 50 تایم‌استپ
    labels = torch.randint(0, 2, (100,))  # خروجی 0 یا 1 برای 100 نمونه
    return inputs, labels


@pytest.fixture
def data_loader(sample_data):
    """ایجاد DataLoader برای مدیریت داده‌های تست"""
    inputs, labels = sample_data
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=16)


def test_cnn_model(data_loader):
    """تست مدل CNN"""
    model = CNNModel(input_channels=3, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # آموزش یک اپوک
    model.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # بررسی خروجی مدل
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(data_loader))
        outputs = model(inputs)
        assert outputs.shape == (16, 2), "ابعاد خروجی مدل CNN اشتباه است."


def test_lstm_model(data_loader):
    """تست مدل LSTM"""
    model = LSTMModel(input_size=3, hidden_size=64, num_layers=2, output_size=1)  # تغییر input_size به 50
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # آموزش یک اپوک
    model.train()
    for inputs, labels in data_loader:
        inputs = inputs.permute(0, 2, 1)  # تغییر شکل داده‌ها به (batch_size, sequence_length, input_size)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

    # بررسی خروجی مدل
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(data_loader))
        inputs = inputs.permute(0, 2, 1)  # تغییر شکل داده‌ها به (batch_size, sequence_length, input_size)
        outputs = model(inputs)
        assert outputs.shape == (16, 1), "ابعاد خروجی مدل LSTM اشتباه است."

def test_transformer_model(data_loader):
    """تست مدل Transformer"""
    model = TransformerModel(input_size=50, num_heads=2, num_layers=2, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # آموزش یک اپوک
    model.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # بررسی خروجی مدل
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(data_loader))
        outputs = model(inputs)
        assert outputs.shape == (16, 2), "ابعاد خروجی مدل Transformer اشتباه است."