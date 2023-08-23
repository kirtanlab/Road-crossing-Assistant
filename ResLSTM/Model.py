import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
device = torch.device('cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize LSTM hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass through fully connected layer
        out = self.fc(out[:, -1, :])  # Take the last LSTM output
        
        return out