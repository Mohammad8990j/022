import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.lstm_model import LSTMModel
from src.utils.logger import TradingBotLogger
# تعریف کلاس Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length, :-1]
        y = self.data[index + self.sequence_length, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(data_path, sequence_length=50, epochs=20, batch_size=32, learning_rate=0.001):
    logger = TradingBotLogger("LSTMModel")
    logger.info("شروع آموزش مدل LSTM...")

    # بارگذاری داده‌ها
    df = pd.read_parquet(data_path)
    data = df.values  # Convert to numpy array
    logger.info(f"داده‌ها بارگذاری شدند: {data.shape}")

    # تقسیم داده‌ها به آموزش و تست
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # مدل
    input_size = train_data.shape[1] - 1  # تعداد ویژگی‌ها به غیر از ستون هدف
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # معیار خطا و بهینه‌ساز
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # آموزش مدل
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(model.device), y.to(model.device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")

    # ذخیره مدل
    save_path = Path("saved_models/lstm_model.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"مدل ذخیره شد: {save_path}")

if __name__ == "__main__":
    train_model(data_path="data/processed/EURUSD_H1_processed_20250416_0533.csv")