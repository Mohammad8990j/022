import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.cnn_model import train_cnn_model
from src.models.lstm_model import train_lstm_model
from src.models.transformer_model import train_transformer_model

# مسیر ذخیره‌سازی مدل‌ها
SAVE_DIR = "saved_models/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# تولید داده نمونه (برای آزمایش)
def generate_sample_data(num_samples=1000, sequence_length=20, input_features=3, num_classes=3):
    """
    تولید داده‌های نمونه برای آموزش مدل‌ها.
    """
    X = np.random.rand(num_samples, input_features, sequence_length).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

# آماده‌سازی داده‌ها
def prepare_data(batch_size=32):
    """
    آماده‌سازی DataLoader برای داده‌های نمونه.
    """
    X, y = generate_sample_data()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))  # تبدیل labels به long
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# آموزش مدل‌ها
def main():
    """
    آموزش مدل‌های CNN، LSTM، و Transformer.
    """
    print("آماده‌سازی داده‌ها...")
    train_loader = prepare_data()

    print("آموزش مدل CNN...")
    train_cnn_model(
        train_loader=train_loader,
        input_channels=3,  # تعداد ویژگی‌های ورودی
        output_size=3,  # تعداد کلاس‌ها
        epochs=10,
        learning_rate=0.001,
        save_path=os.path.join(SAVE_DIR, "cnn_model.pt")
    
    )
    print("آموزش مدل LSTM...")
    train_lstm_model(
         train_loader=train_loader,
         input_size=3,  # تعداد ویژگی‌های ورودی (تطبیق با داده‌ها)
         hidden_size=64,
         num_layers=2,
         output_size=1,  # تعداد خروجی‌ها
         epochs=10,
         learning_rate=0.001,
         save_path=os.path.join(SAVE_DIR, "lstm_model.pt")

    )

    print("آموزش مدل Transformer...")
    train_transformer_model(
        train_loader=train_loader,
        input_size=20,  # طول توالی ورودی
        num_heads=2,
        num_layers=2,
        output_size=3,  # تعداد کلاس‌ها
        epochs=10,
        learning_rate=0.001,
        save_path=os.path.join(SAVE_DIR, "transformer_model.pt")
    )

    print("آموزش مدل‌ها به پایان رسید و مدل‌ها ذخیره شدند.")

if __name__ == "__main__":
    main()