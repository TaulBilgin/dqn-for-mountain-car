import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dqn_input(state, pos_space, vel_space):
    
    state_p = np.digitize((state[0]), pos_space)
    state_v = np.digitize((state[1]), vel_space)
    input_tensor = torch.Tensor([state_p, state_v]).to(device)
     
    return input_tensor


def optimize(memory, policy_dqn, pos_space, vel_space, learnin_rate, gamma, finis):
    # Sample a batch of transitions from memory
    random_memoy = random.sample(memory, 100)

    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=learnin_rate)
    loss_fn = nn.MSELoss()

    current_q_list = []
    target_q_list = []

    # Iterate over each transition in the sampled batch
    for now_state, action, new_state, reward, done in random_memoy:
        # Assign a high reward if the episode is finished successfully
        if done: 
            target = 10
        else:
            if finis : 
                reward = 1 # if the episode is finished successfully, the all steps take reward
            with torch.no_grad():
                target = reward + gamma * policy_dqn(dqn_input(new_state, pos_space, vel_space)).max().item()
        
        # Get the current Q-value
        current_q = policy_dqn(dqn_input(now_state, pos_space, vel_space))
        current_q_list.append(current_q)
        
        # Create a copy of the current Q-values for updating
        target_q = current_q.clone()
        target_q[action] = target # Update the Q-value for the taken action
        target_q_list.append(target_q)

    # Compute the loss between current and target Q-values
    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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

def save_test(policy_dqn, pos_space, vel_space):
    env2 = gym.make('MountainCar-v0')
    
    now_state = env2.reset()[0]  # Reset environment and get initial state
    done = False  # Flag to check if the episode is finished
    step = 0
    
    # Play one episode
    while (not done) and step < 2000 :
        # Use the policy network to select the best action
        with torch.no_grad():
            action = policy_dqn(dqn_input(now_state, pos_space, vel_space)).argmax().item()  # Best action
        step += 1

        # Take action and observe result
        new_state, reward, done, truncated, _ = env2.step(action)
        
        # Move to the next state
        now_state = new_state
    env2.close()

    return done # Return True if the episode finished successfully, otherwise False

    
def train ():
    env = gym.make('MountainCar-v0')

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)  # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)  # Between -0.07 and 0.07
    
    in_states = 2
    h1_nodes = 16
    h2_nodes = 8
    out_actions = 3

    policy_dqn = DQN(in_states, h1_nodes, h2_nodes, out_actions).to(device)
    learning_rate = 0.01
    gamma = 0.9

    # Initialize the epsilon-greedy action selection choices
    choice_list = ['x'] * 30 + ['y'] * 70

    memory = []
    run = 0
    real_run = 0  # real_run mean to "the episode finished successfully"
    past_real_run = 0 
    best_step = 1000
    

    while True :
        now_state = env.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0
        finis = False
        run += 1  # Increment the episode counte

        # Play one episode
        while (not done) and step < 1000 :
            # Use the policy network to select the best action
            if random.choice(choice_list) == "x":
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():
                    action = policy_dqn(dqn_input(now_state, pos_space, vel_space)).argmax().item()  # Best action
            step += 1

            # Take action and observe result
            new_state, reward, done, truncated, _ = env.step(action)

            # Store transition in memory
            memory.append((now_state, action, new_state, reward, done))

            now_state = new_state
            
        percentage = (real_run * 100) / (run)
        print(f"run : {run} | real run : {real_run} | % : {percentage}")

        # Conditions to stop training if performance criteria are not met
        if real_run > 1000 and percentage < 45:
            return 1
        
        if (run > 100 and real_run < 25) or (run > 500 and percentage < 40 ) :
            return 0
        
        # Update the real run count if the episode finished successfully
        if done :
            real_run += 1
            finis = True
        optimize(memory, policy_dqn, pos_space, vel_space, learning_rate, gamma, finis)
        
        # update the choice list based on the successfully 50 run
        if (real_run % 50) == 0 and  past_real_run != real_run :
            past_real_run = real_run
            if all(choice == 'y' for choice in choice_list):
                torch.save(policy_dqn.state_dict(), "mountion_car_dql.pt")
                return 1

            choice_list.remove("x")
            choice_list.append("y")

        # Save the best model based on steps taken
        if step < best_step and real_run > 1000 :
            env.close()
            t_or_f = save_test(policy_dqn, pos_space, vel_space)
            if t_or_f :
                best_step = step
                torch.save(policy_dqn.state_dict(), f"mountion_car_dql_{best_step}.pt")
        memory = []

while True:
    policy_dqn = train()
    if policy_dqn == 1 :
        break
    