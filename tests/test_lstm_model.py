import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from src.models.lstm_model import LSTMModel

def test_lstm_model(data_loader):
    """تست مدل LSTM"""
    model = LSTMModel(input_size=50, hidden_size=64, num_layers=2, output_size=1)  # تغییر input_size به 50
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