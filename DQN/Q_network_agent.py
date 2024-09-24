# DQN agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from Connect4 import Connect4
import time

# Define the Q-network
class QNetwork(nn.Module):
    # input just a state and output is q value for all possible actions
    # input = state_size + player turn = 8x8 + 8x1 = 72
    # player0 turn = [0, 0, 0, 0, 0, 0, 0, 0]
    # player1 turn = [1, 1, 1, 1, 1, 1, 1, 1]
    # input[8] = playerx turn 
    # output = number of action choices for every step = number of cols in the board = 8

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
q_network = QNetwork(64, 7)  # Use the same architecture used during training
q2_network = QNetwork(64, 7)  # bigger training episodes

# Load the saved parameters into the model
q_network.load_state_dict(torch.load('q_network.pth'))

q2_network.load_state_dict(torch.load('q5k.pth'))
# Put the model in evaluation mode (disables dropout, etc.)
q_network.eval()


env = Connect4()

player_turn = 0
done = False

while not done:
    time.sleep(1)
    player_turn += 1
    player_turn = player_turn % 2 # make sure its -1 to 1
    env.set_player_turn(player_turn)

    # get connect4 board state and then append player label 
    board_state = env.get_board_state()
    #print(type(board_state), board_state)
    player_tensor = torch.tensor([player_turn]*22)
    state = torch.cat((board_state, player_tensor))  

    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        if player_turn %2 == 0:
            q_values = q2_network(state)
            action = np.argmax(q_values.cpu().data.numpy())
            print(q_values, action)
        else:
            #q_values = q_network(state)  # playing against another agent
            action = int(input("Player 2, make your selection (0-7): "))  # choose col index 

      # best action based on max Qvalue (col index)
    #print('DDDD', action, env.step(action))

    # get a new state based on the action and reward in the new state.

    next_state, reward, done = env.step(action)  # calling Connect4.step
    next_state = torch.from_numpy(next_state).float()


    # Flatten the grid to feed it into a fully connected network
    next_state = next_state.view(-1)  # Shape becomes (64+8,)
    next_state = torch.cat((next_state, player_tensor))  # adding player tensor

    env.print_board()
    


