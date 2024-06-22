import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dqn_input(state, pos_space, vel_space):
    
    state_p = np.digitize((state[0]), pos_space)
    state_v = np.digitize((state[1]), vel_space)
    input_tensor = torch.Tensor([state_p, state_v]).to(device)
     
    return input_tensor

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super(DQN, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_actions)

    def forward(self, x):
        # Define the forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    

# Create the MountainCar environment with rendering enabled
env = gym.make('MountainCar-v0', render_mode="human")

# Divide position and velocity into segments
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07

# Define the neural network parameters
in_states = 2
h1_nodes = 16 
h2_nodes = 8
out_actions = 3 

# Initialize the DQN model
policy_dqn = DQN(in_states, h1_nodes, h2_nodes, out_actions).to(device)

# Load the trained model weights
policy_dqn.load_state_dict(torch.load("your model name "))

# Switch the model to evaluation mode
policy_dqn.eval()

# Initialize counters
run = 0
real_run = 0

# Reset the environment and get the initial state
now_state = env.reset()[0]
done = False  # Flag to check if the episode is finished
step = 0
run += 1  # Increment the episode counter

# Play one episode
while not done and step < 200:
    # Use the policy network to select the best action
    with torch.no_grad():
        action = policy_dqn(dqn_input(now_state, pos_space, vel_space)).argmax().item()
    
    step += 1  # Increment step counter

    # Take action and observe the result
    new_state, reward, done, truncated, _ = env.step(action)
    
    # Move to the next state
    now_state = new_state
    
    
print(f"run : {run} | real run : {real_run} | % : {(real_run * 100) / (run)}")



