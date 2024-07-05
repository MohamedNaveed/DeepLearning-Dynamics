import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


import pandas as pd
from sklearn.model_selection import train_test_split

from network_models import RK4Net #ResNet

device = ("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

print(f"Using {device} device")

model = RK4Net().to(device)
print(model)

if __name__ == "__main__":
    exp_name = 'diffInitialConditions'
    data = pd.read_csv('../data/pendulum_exps/'+exp_name+'/traindata_1M.csv')
    dim = 2
    print(data.head()) 

    # Extract input features (X) and target labels (y)
    X = data[['Pendulum Angle (rad)', 'Angular Velocity (rad/s)']].values  # Input features
    y = data[['Pendulum Angle next (rad)', 'Angular Velocity next (rad/s)']].values  # Target labels
    
    print(X[0],y[0])

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Size of training = ', X_train.shape[0]);
    print('Size of testing = ', X_test.shape[0]);

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
    
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
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

    torch.save(model.state_dict(), '../models/'+'pendulum_trained_1M_rk4net_'+ exp_name +'.pth')
    print("Saved PyTorch Model State to pendulum_trained.pth")