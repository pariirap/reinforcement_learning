# DQN agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tweetybird import TweetyBird

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
    

# Hyperparameters
BUFFER_SIZE = 1000     # Replay buffer size
BATCH_SIZE = 64         # Mini-batch size
GAMMA = 0.99            # Discount factor
LR = 0.001              # Learning rate
TARGET_UPDATE = 50      # Target network update frequency
EPSILON_START = 1.0     # Initial epsilon for exploration
EPSILON_END = 0.01      # Final epsilon
EPSILON_DECAY = 0.99   # Epsilon decay rate

DEBUG = True  # just to see the print statement


class DQNAgentTweety:

    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START

        # Q-network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()   # not really applicable for now - jyo 

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        # Replay memory
        self.memory = deque(maxlen=BUFFER_SIZE)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return np.argmax(q_values.cpu().data.numpy())  # choosing max action by taking h index of max q_values

    def act_random(self, state):  # for INVALID MOVES
        return random.randint(0, self.action_size - 1)
        
        

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
       
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states  = torch.vstack(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        next_states = torch.vstack(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
    
        q_values = self.q_network(states).gather(1, actions)  
        
        # Target Q values
        with torch.no_grad(): #
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)  # swapping target_network 
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)
        if DEBUG and self.epsilon < 0.2 and random.randint(0, 1000) > 900:
            #print(f"Loss: {loss}, Q: {q_values}, T: {target_q_values}")
            print(f"Loss: {loss}")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())



# Training loop
def train_dqn(env: TweetyBird, agent, num_episodes):

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        player_turn = 0
        count = 0
        while not done and count < 3:

            # get connect4 board state and then append player label 
            state = env.get_board_state()
            #print(type(board_state), board_state)
            
            action = agent.act(state)  # best action based on max Qvalue (col index)
            #print('DDDD', action, env.step(action))
            # get a new state based on the action and reward in the new state.
            next_state, reward, done = env.step(action, count)  # calling tweety.step

            next_state = torch.from_numpy(next_state).float()

            # Flatten the grid to feed it into a fully connected network
            next_state = next_state.view(-1)  # Shape becomes (64+8,)
            
            # pushing into replay buffer
            agent.remember(state, action, reward, next_state, done)

            # state = next_state
            total_reward += reward
            count += 1

            # gradient update.
            agent.replay()


           
        # # Decrease epsilon
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")


    # save the agent
    torch.save(agent.q_network.state_dict(), 'q2k.pth')




env = TweetyBird()
state_size = 9 # 3x3 grid
action_size = 3
print(state_size, action_size)
agent = DQNAgentTweety(state_size, action_size)
train_dqn(env, agent, num_episodes=3000)