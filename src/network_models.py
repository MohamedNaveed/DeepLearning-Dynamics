
import torch 
from torch import nn

# Define model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
class ResNet(MLP):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) + x
        return logits
    
class RK4Net(MLP):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = self.flatten(x)
        k1 = self.linear_relu_stack(x)
        k2 = self.linear_relu_stack(x + k1/2.0)
        k3 = self.linear_relu_stack(x + k2/2.0)
        k4 = self.linear_relu_stack(x + k3)
    
        logits = x + (k1 + 2*k2 + 2*k3 + k4)/6.0
    
        return logits
    

class LSTMNet(nn.Module):
    def __init__(self, batch_size=256, device="cuda"):
        super().__init__()
        self.input_size = 2
        self.hidden_size = 512
        self.output_size = 2
        self.batch_size = batch_size
        self.device = device
        self.num_layers = 2

        self.rnn = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            batch_first = True
            )
        self.linear = nn.Linear(self.hidden_size,self.output_size)

    
    def forward(self, x):

        # Forward pass through LSTM
        output, (hn, cn) = self.rnn(x)
        output = self.linear(output)

        return output

