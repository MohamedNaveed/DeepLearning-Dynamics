import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from network_models import LSTMNet #ResNet RK4Net



device = ("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

print(f"Using {device} device")

network = 'lstm'
exp_name = 'diffInitialConditions'
angle = '15deg'
file_name = f'testdata_sequence_{angle}.csv'

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
    batch_size = 32 # mini batch size
     
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
    
    print('input sequence = ', input_sequences[0])
    print('output sequence = ', output_sequences[0])
    # Create dataset
    #dataset = PendulumDataset(input_sequences, output_sequences)

    # Split the data into training and testing sets (80% train, 20% test)
    X = input_sequences
    y = output_sequences
    
    target_array = y
    n_rows = X.shape[0]
    print('n_rows=',n_rows)

    # Convert NumPy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    test_dataset = TensorDataset(X_tensor, y_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()

    n_samples = n_rows + seq_length 
    predicted_1_step = np.zeros((n_samples,2))
    true_1_step = np.zeros((n_samples,2))

    

    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(test_dataloader):
            
            if batch == 0:
                #set initial conditions
                predicted_1_step[0:seq_length,:] = inputs[0,:,:]
                true_1_step[0:seq_length,:] = inputs[0,:,:]
                #print(f"inputs = {inputs[0,:,:]}, targets = {targets[0,:,:]}")
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outs_np = outputs.squeeze().cpu().detach().numpy()
            #print(outs_np.shape)
            
            outputs_size = outs_np.shape[0]

            start_idx = seq_length + batch*batch_size
            
            if outputs_size == batch_size:
                end_idx = start_idx + batch_size
            else:
                end_idx = start_idx + outputs_size
                
            predicted_1_step[start_idx:end_idx,:] = outs_np[:,-1,:]
            
            true_1_step[start_idx:end_idx,:] = targets.cpu().detach().numpy()[:,-1,:]

            test_loss += criterion(outputs.squeeze(), targets).item()
            
    test_loss /= len(test_dataloader)
    print(f'1 step Test Loss: {test_loss:.8f}')

    # Evaluation using recursive predictions. 
    model.eval()

    dt = 0.01
    t_span = n_samples*dt
    t_steps = n_samples - seq_length # int(t_span/dt) #+1 to include time step =0
    seq_outputs_array = np.zeros((n_samples,2))
    des_outputs_array = np.zeros((n_samples,2))
    print('target shape=', target_array.shape)

    #set initial condition.
    seq_outputs_array[0:seq_length,:] = true_1_step[0:seq_length,:] 
    des_outputs_array[0:seq_length,:] = true_1_step[0:seq_length,:]
    test_loss = 0.0

    with torch.no_grad():
        for i in range(target_array.shape[0]):
            
            input = torch.tensor(seq_outputs_array[i:seq_length+i,:], 
                                 dtype=torch.float32).view(1,seq_length,seq_outputs_array.shape[1])
            target = torch.tensor(target_array[i,-1,:], dtype=torch.float32).view(1,target_array.shape[2])
            #print('predicted:', seq_outputs_array[i:seq_length+i,:], 'truth:',target_array[i+1,-1,:])
            
            #print("target shape = ", target.shape, " input shape = ", input.shape)
            input, target = input.to(device), target.to(device)
            
            output = model(input)
            #print('output shape = ', output.shape)
            out_np = output.squeeze().cpu().detach().numpy()
            seq_outputs_array[seq_length+i,:] = out_np[-1,:]
            des_outputs_array[seq_length+i,:] = target_array[i,-1,:]
            #print(f'Output of network = {out_np[-1,:]}; target = {target_array[i,-1,:]}')
            
            test_loss += criterion(output.squeeze()[-1,:], target.squeeze()).item()
            #print(f'i = {i}; Test loss = {test_loss}')
            
    test_loss /= t_steps
    print(f'Recursive Test Loss (per step): {test_loss:.8f}')

    t = np.linspace(0,t_span,n_samples)
    
    # Create subplots for angle and angular velocity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot pendulum angle
    ax1.plot(t, des_outputs_array[:,0], label='True Pendulum Angle (rad)', color='orange')
    ax1.plot(t, seq_outputs_array[:,0], '--', label='Predicted Pendulum Angle (rad)',color='blue')

    ax1.set_title('Pendulum Response')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (rad)')
    ax1.grid(True)
    ax1.legend()

    # Plot angular velocity
    ax2.plot(t, des_outputs_array[:,1], label='True Angular Velocity (rad/s)', color='orange')
    ax2.plot(t, seq_outputs_array[:, 1], '--',label='Predicted Angular Velocity (rad/s)', color='blue')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    fig.savefig(f'../results/pendulum_exps/{exp_name}/{network}_seq_prediction_{angle}.png', bbox_inches='tight')


    t = np.linspace(0,n_samples*dt,n_samples)
    # Create subplots for angle and angular velocity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot pendulum angle
    ax1.plot(t, true_1_step[:, 0], label='True Pendulum Angle (rad)', color='orange')
    ax1.plot(t, predicted_1_step[:, 0], '--', label='Predicted Pendulum Angle (rad)',color='blue')

    ax1.set_title('Pendulum Response')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (rad)')
    ax1.grid(True)
    ax1.legend()

    # Plot angular velocity
    ax2.plot(t, true_1_step[:, 1], label='True Angular Velocity (rad/s)', color='orange')
    ax2.plot(t, predicted_1_step[:, 1], '--',label='Predicted Angular Velocity (rad/s)', color='blue')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    fig.savefig(f'../results/pendulum_exps/{exp_name}/{network}_1step_prediction_{angle}.png', bbox_inches='tight')