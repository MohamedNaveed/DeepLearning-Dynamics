import math as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class pendulum():
    def __init__(self,x0, n_samples=1000):
        self.x0 = x0
        self.dim = 2
        self.dt = 0.01 #discretization time
        self.n_samples = n_samples #total time
        
    def pendulum_ode(self, x, t):
        
        x1_dot = x[1]
        x2_dot = -m.sin(x[0])

        dxdt = np.array([x1_dot,x2_dot])

        return dxdt
        
    def simulate_data(self):

        x = np.zeros((2,self.n_samples), dtype=float)
        x[:,0] = self.x0.reshape((2,))
        t = [0]

        for i in range(self.n_samples-1):

            x[:,i+1] = self.RK4(self.pendulum_ode, x[:,i], t[-1], self.dt).reshape((2,))
            t.append(t[-1] + self.dt)

        return t,x

    def RK4(self, odefunc,x0,t0,h):

        k1 = h*odefunc(x0,t0)
        k2 = h*odefunc(x0 + k1/2, t0 + h/2)
        k3 = h*odefunc(x0 + k2/2, t0 + h/2)
        k4 = h*odefunc(x0 + k3, t0 + h)
    
        x = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
    
        return x

    def plot_response(self, t, states):
        
        # Create subplots for angle and angular velocity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Plot pendulum angle
        ax1.plot(t, states[0,:], label='Pendulum Angle (rad)')
        ax1.set_title('Pendulum Response')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Angle (rad)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot angular velocity
        ax2.plot(t, states[1,:], label='Angular Velocity (rad/s)', color='orange')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.grid(True)
        ax2.legend()
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()


def write_to_csv(states,dim,path):

    dataset_size = states.shape[1]
    #print('dataset_size=', dataset_size)

    data = pd.DataFrame({
    'Pendulum Angle (rad)': states[0,0:dataset_size-1],
    'Angular Velocity (rad/s)': states[1,0:dataset_size-1],
    'Pendulum Angle next (rad)': states[0,1:dataset_size],
    'Angular Velocity next (rad/s)': states[1,1:dataset_size],})

    csv_file_path = path

    # Check if the file already exists
    try:
        # Read existing data from the CSV file (if it exists)
        existing_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file doesn't exist, create a new file with the data
        data.to_csv(csv_file_path, index=False)
    else:
        # Append the new data to the existing file
        data.to_csv(csv_file_path, mode='a', header=False, index=False)



    

if __name__=='__main__':

    # Generate random initial conditions uniformly
    num_conditions = 1
    theta0_values = np.random.uniform(low=-(np.pi/2), high=np.pi/2, size=num_conditions)
    #omega0_values = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=num_conditions)
     
    #theta0_values[0] = np.pi/2

    for n in range(num_conditions):
        system = pendulum(np.array([[theta0_values[n]], [0]]),n_samples=10000)
        [timesteps, states] = system.simulate_data()
        #system.plot_response(timesteps, states)

        path = '/home/naveed/Documents/deep_learning/data/pendulum_exps/near90deg/testdata_90.csv'
        write_to_csv(states, system.dim, path)

