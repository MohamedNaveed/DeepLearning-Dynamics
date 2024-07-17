import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from network_models import LSTMNet #ResNet RK4Net

device = ("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

print(f"Using {device} device")

network = 'lstm'
exp_name = 'diffInitialConditions'
file_name = 'testdata_sequence_90deg.csv'

model = LSTMNet(device = device).to(device)
print(model)
# load pretrained model.
model_path = f"../models/pendulum_trained_{network}_{exp_name}.pth"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

    
if __name__ == "__main__":

    
    df = pd.read_csv(f'../data/pendulum_exps/{exp_name}/{file_name}')
    dim = 2
    print(df.head()) 

    # Parameters
    seq_length = 10
    n_features = 2  # Number of features (Angle and AngularVelocity)

    # Convert DataFrame rows back to sequences
    sequences = []

    for row in df.itertuples(index=False):
        data = np.array(row)
        input_seq = data[:seq_length * n_features].reshape((seq_length, n_features))
        output_seq = data[seq_length * n_features:].reshape((seq_length, n_features))
        sequences.append((input_seq, output_seq))

    # Verify the sequence
    #print("sequences = ", sequences[0])

    # Convert sequences to PyTorch tensors
    input_sequences = np.array([seq[0] for seq in sequences])
    output_sequences = np.array([seq[1] for seq in sequences])

    
    # Verify the tensor shape
    print("input seq shape=", input_sequences[0].shape)
    print("output seq shape=", output_sequences[0].shape)

    # Create dataset
    #dataset = PendulumDataset(input_sequences, output_sequences)

    # Split the data into training and testing sets (80% train, 20% test)
    X = input_sequences
    y = output_sequences
    
    target_array = y
    n_samples = X.shape[0]
    print('n_samples=',n_samples)
    # Convert NumPy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    test_dataset = TensorDataset(X_tensor, y_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluation
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()

    outputs_array = np.zeros((n_samples,2))

    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(test_dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outs_np = outputs.squeeze().cpu().detach().numpy()

            outputs_size = outs_np.shape[0]
            outputs_array[batch*outputs_size:(batch+1)*outputs_size,:] = outs_np
            
            test_loss += criterion(outputs.squeeze(), targets).item()
            
    test_loss /= len(test_dataloader)
    print(f'1 step Test Loss: {test_loss:.8f}')

    # Evaluation using recursive predictions. 
    model.eval()

    dt = 0.01
    t_span = n_samples*dt
    t_steps = n_samples # int(t_span/dt) #+1 to include time step =0
    seq_outputs_array = np.zeros((t_steps,2))
    seq_outputs_array[0,:] = target_array[0,:] #set initial condition.
    test_loss = 0.0
    with torch.no_grad():
        for i in range(t_steps-1):
            
            input = torch.tensor(seq_outputs_array[i,:], dtype=torch.float32).view(1,seq_outputs_array.shape[1])
            target = torch.tensor(target_array[i+1,:], dtype=torch.float32).view(1,target_array.shape[1])
            #print('predicted:', seq_outputs_array[i,:], 'truth:',target_array[i,:])
            
            input, target = input.to(device), target.to(device)
            
            output = model(input)

            out_np = output.cpu().detach().numpy()
            seq_outputs_array[i+1,:] = out_np
            #print(f'Output of network = {output}; target = {target}')
            test_loss += criterion(output.squeeze(), target.squeeze()).item()
            #print(f'Test loss = {test_loss}')
            
    test_loss /= t_steps
    print(f'Recursive Test Loss (per step): {test_loss:.8f}')

