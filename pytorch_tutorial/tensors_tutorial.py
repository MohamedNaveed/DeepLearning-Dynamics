import torch
import numpy as np


shape = (1,3,)
data = torch.rand(shape)

print(data)

print(f"Shape of tensor: {data.shape}")
print(f"Datatype of tensor: {data.dtype}")
print(f"Device tensor is stored on: {data.device}")

device = ("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

print(f"Using {device} device")

data = data.to(device)
print(f"Device tensor is stored on: {data.device}")
