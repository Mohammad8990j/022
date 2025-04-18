import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # تعریف لایه LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # لایه Fully Connected برای خروجی
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Cell state
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))  # Output shape: [batch_size, sequence_length, hidden_size]
        
        # فقط آخرین خروجی LSTM را به لایه Fully Connected می‌دهیم
        out = self.fc(out[:, -1, :])  # Output shape: [batch_size, output_size]
        
        return out