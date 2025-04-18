import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from src.models.lstm_model import LSTMModel

def test_lstm_model():
    input_size = 5
    hidden_size = 64
    num_layers = 2
    output_size = 1
    sequence_length = 50
    batch_size = 32

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    x = torch.randn(batch_size, sequence_length, input_size)
    y = model(x)

    assert y.shape == (batch_size, output_size), "خروجی مدل LSTM اشتباه است!"