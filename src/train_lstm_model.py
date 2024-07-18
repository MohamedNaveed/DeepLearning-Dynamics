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
file_name = 'traindata_sequence_100K.csv'

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Size of training = ', len(X_train))
    print('Size of testing = ', len(X_test))
    print('Shape of X_train[0]=', X_train[0].shape, ' target=', y_train[0].shape)

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    print('Shape of X_train_tensor[0]=', X_train_tensor[0].shape)

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
    
            inputs, targets = inputs.to(device), targets.to(device)
            if torch.isnan(inputs).any():
                print("Input contains NaN values")
            if torch.isnan(targets).any():
                print("Targets contain NaN values")

            optimizer.zero_grad()
            
            outputs = model(inputs)
            #print("outputs=",outputs)
            loss = criterion(outputs.squeeze(), targets)
            
            loss.backward() #retain_graph=True
            
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader):.8f}')
    
    print('Training finished!')

    # Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
    
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs.squeeze(), targets).item()
    test_loss /= len(test_dataloader)
    print(f'Test Loss: {test_loss:.8f}')
    
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")