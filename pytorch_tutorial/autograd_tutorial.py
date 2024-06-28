import os
import torch
from torch import nn

x = torch.ones(5) #input
y = torch.zeros(3) # expected output

w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3, requires_grad=True) 

z = torch.matmul(x, w) + b

loss = nn.functional.binary_cross_entropy_with_logits(z,y)

loss.backward()
print(w.grad)
print(b.grad)


