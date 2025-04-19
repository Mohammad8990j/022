import os
import torch
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.cnn_model import train_cnn_model
from src.models.lstm_model import train_lstm_model
from src.models.transformer_model import train_transformer_model

# مسیر ذخیره‌سازی موقت برای تست
TEST_SAVE_DIR = "tests/saved_models/"
if not os.path.exists(TEST_SAVE_DIR):
    os.makedirs(TEST_SAVE_DIR)

# تولید داده نمونه برای تست
def generate_sample_data(num_samples=100, sequence_length=20, input_features=3, num_classes=3):
    X = np.random.rand(num_samples, input_features, sequence_length).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

# آماده‌سازی داده‌ها
def prepare_test_data(batch_size=32):
    X, y = generate_sample_data()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# تست مدل CNN
def test_cnn_model():
    train_loader = prepare_test_data()
    model_path = os.path.join(TEST_SAVE_DIR, "cnn_model.pt")
    model = train_cnn_model(
        train_loader=train_loader,
        input_channels=3,
        output_size=3,
        epochs=1,  # آموزش سریع برای تست
        learning_rate=0.001,
        save_path=model_path
    )
    assert os.path.exists(model_path), "مدل CNN ذخیره نشد!"
    print("تست مدل CNN با موفقیت انجام شد.")

# تست مدل LSTM
def test_lstm_model():
    train_loader = prepare_test_data()
    model_path = os.path.join(TEST_SAVE_DIR, "lstm_model.pt")
    model = train_lstm_model(
        train_loader=train_loader,
        input_size=3,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        epochs=1,  # آموزش سریع برای تست
        learning_rate=0.001,
        save_path=model_path
    )
    assert os.path.exists(model_path), "مدل LSTM ذخیره نشد!"
    print("تست مدل LSTM با موفقیت انجام شد.")

# تست مدل Transformer
def test_transformer_model():
    train_loader = prepare_test_data()
    model_path = os.path.join(TEST_SAVE_DIR, "transformer_model.pt")
    model = train_transformer_model(
        train_loader=train_loader,
        input_size=20,
        num_heads=2,
        num_layers=2,
        output_size=3,
        epochs=1,  # آموزش سریع برای تست
        learning_rate=0.001,
        save_path=model_path
    )
    assert os.path.exists(model_path), "مدل Transformer ذخیره نشد!"
    print("تست مدل Transformer با موفقیت انجام شد.")

if __name__ == "__main__":
    test_cnn_model()
    test_lstm_model()
    test_transformer_model()