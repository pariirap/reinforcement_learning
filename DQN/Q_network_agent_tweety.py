# DQN agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tweetybird import TweetyBird
import time

# Define the Q-network
class QNetwork(nn.Module):
    # input just a state and output is q value for all possible actions
    

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
    

#Initialize the model structure
q_network = QNetwork(9, 3)  # Use the same architecture used during training

q_network.load_state_dict(torch.load('q2k.pth'))


# Put the model in evaluation mode (disables dropout, etc.)
q_network.eval()


env = TweetyBird()

player_turn = 0
done = False

count = 0
while not done:
    time.sleep(1)
    
    # get connect4 board state and then append player label 
    board_state = env.get_board_state()
    #print(type(board_state), board_state)
    state = torch.FloatTensor(board_state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state)
        action = np.argmax(q_values.cpu().data.numpy())
        print(q_values, action)
    
    # get a new state based on the action and reward in the new state.
    next_state, reward, done = env.step(action, count)  # calling tweety.step
    next_state = torch.from_numpy(next_state).float()

    # Flatten the grid to feed it into a fully connected network
    next_state = next_state.view(-1)  # Shape becomes (64+8,)
    
    env.print_board()
    count += 1 
    


