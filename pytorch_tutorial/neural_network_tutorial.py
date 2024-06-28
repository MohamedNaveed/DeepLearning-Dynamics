import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

#find device
device = ("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

print(f"using device {device}")

input_size = 28*28 
output_size = 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(f'Neural network model {model}')

X = torch.rand(3,28,28, device=device)

output = model(X)
print(output)

pred_prob = nn.Softmax(dim=1)(output)
print(pred_prob)

y_pred = pred_prob.argmax(1)
print(f'Predicted class {y_pred}')