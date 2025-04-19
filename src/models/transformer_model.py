import torch
import torch.nn as nn
import torch.optim as optim


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.fc(x)
        return x


def train_transformer_model(train_loader, input_size, num_heads, num_layers, output_size, epochs=20, learning_rate=0.001, save_path="saved_models/transformer_model.pt"):
    model = TransformerModel(input_size, num_heads, num_layers, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # ذخیره مدل
    torch.save(model.state_dict(), save_path)
    print(f"مدل Transformer در مسیر {save_path} ذخیره شد.")

    return model