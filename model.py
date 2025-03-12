import torch
import torch.nn as nn

class VoiceIDModel(nn.Module):
    def __init__(self, input_dim=13, num_classes=1):
        super(VoiceIDModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def train_model(model, data_loader, device, epochs=10, learning_rate=0.001):
    """
    Placeholder training loop.
    Training on GPU will be used if device is set accordingly.
    """
    model.to(device)
    criterion = nn.MSELoss()  # Adjust as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for features, target in data_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed")
    return model

def infer(model, feature, device):
    """
    Runs inference on a single feature vector.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.tensor(feature, dtype=torch.float32).to(device)
        output = model(feature_tensor)
    return output.cpu().numpy()
