import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
device = torch.device('cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,layers=2,batch_first=True)

    def forward(self, x):   
        # Pass through LSTM
        out, _ = self.lstm(x)
        out = out[:,-1,:]  # Take the last LSTM output
        return out