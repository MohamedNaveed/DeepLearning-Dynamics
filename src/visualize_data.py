import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

exp_name = 'diffInitialConditions'

def plot_trajectory(t, X):

    # Create subplots for angle and angular velocity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot pendulum angle
    ax1.plot(t, X[:, 0], label='True Pendulum Angle (rad)', color='orange')

    ax1.set_title('Pendulum Response')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (rad)')
    ax1.grid(True)
    ax1.legend()

    # Plot angular velocity
    ax2.plot(t, X[:, 1], label='True Angular Velocity (rad/s)', color='orange')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_phase_portrait(X):

    fig = plt.figure(figsize=(8, 10))
    plt.plot(X[:,0], X[:,1],'o', markersize=3, color='blue',alpha=0.7)
    plt.title('Phase portrait')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid(True)
    fig.savefig('../results/pendulum_exps/'+ exp_name +'/phasePortrait.png', bbox_inches='tight')
    #plt.show()

    


if __name__ == '__main__':

    data = pd.read_csv('../data/pendulum_exps/diffInitialConditions/traindata.csv')

    # Extract input features (X) and target labels (y)
    X = data[['Pendulum Angle (rad)', 'Angular Velocity (rad/s)']].values  # Input features

    n_samples = X.shape[0]
    print('n_samples=',n_samples)
    t = np.linspace(0,n_samples*0.01,n_samples)
    plot_phase_portrait(X)