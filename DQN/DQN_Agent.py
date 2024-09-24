# DQN agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from Connect4 import Connect4

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
BATCH_SIZE = 128         # Mini-batch size
GAMMA = 0.99            # Discount factor
LR = 0.001              # Learning rate
TARGET_UPDATE = 50      # Target network update frequency
EPSILON_START = 1.0     # Initial epsilon for exploration
EPSILON_END = 0.01      # Final epsilon
EPSILON_DECAY = 0.9995   # Epsilon decay rate

DEBUG = True  # just to see the print statement


class DQNAgent:

    
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
        

    # def step_(self, action, gym):
        
    #     '''
    #     call connect4 and get new state and reward if any and if done. 

    #     '''
    #     new_state, reward, done = gym.step(action)

    #     return new_state, reward, done
        

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        try:
            batch = random.sample(self.memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            #states = torch.FloatTensor(states)
            #states = torch.tensor(states, dtype=torch.float32)
            states  = torch.vstack(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            #next_states = torch.FloatTensor(next_states)
            next_states = torch.vstack(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)
        


        
        #q2 = self.q_network(states[1])
        # Current Q values
        # we are not storing q_values, because the network is changing. only s, a, r, s'
            q_values = self.q_network(states).gather(1, actions)  
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
        # Target Q values
        with torch.no_grad(): #
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)  # swapping target_network 
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)
        if DEBUG and self.epsilon < 0.2 and random.randint(0, 10000) > 9990:
            #print(f"Loss: {loss}, Q: {q_values}, T: {target_q_values}")
            print(f"Loss: {loss}")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # Decrease epsilon
        # if self.epsilon > EPSILON_END:
        #     self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())



# Training loop
def train_dqn(env: Connect4, agent, num_episodes):

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        player_turn = 0
        count = 0
        while not done:
            player_turn += 1
            player_turn = player_turn % 2 # make sure its -1 to 1
            env.set_player_turn(player_turn)

            # get connect4 board state and then append player label 
            board_state = env.get_board_state()
            #print(type(board_state), board_state)
            player_tensor = torch.tensor([player_turn]*22)
            state = torch.cat((board_state, player_tensor))  

            
            
            action = agent.act(state)  # best action based on max Qvalue (col index)
            #print('DDDD', action, env.step(action))
            # get a new state based on the action and reward in the new state.
            next_state, reward, done = env.step(action)  # calling Connect4.step

            if reward == -10:
                INVALID_MOVE = True
                while INVALID_MOVE:
                    action = agent.act_random(state)
                    next_state, reward, done = env.step(action)  # calling Connect4.step
                    if reward != -10:
                        INVALID_MOVE =False
                
            next_state = torch.from_numpy(next_state).float()


            # Flatten the grid to feed it into a fully connected network
            next_state = next_state.view(-1)  # Shape becomes (64+8,)
            next_state = torch.cat((next_state, player_tensor))  # adding player tensor

            # pushing into replay buffer
            agent.remember(state, action, reward, next_state, done)

            # state = next_state
            total_reward += reward
            count += 1

            # gradient update.
            agent.replay()


            #if count % 100 == 0:
                #print(env.print_board())
                #print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        # # Decrease epsilon
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")


    # save the agent
    torch.save(agent.q_network.state_dict(), 'q8k.pth')




env = Connect4()
state_size = 64 # 42+22 =  6x7 grid and 22 duplicate  player label for enforcement
action_size = 7
print(state_size, action_size)
agent = DQNAgent(state_size, action_size)
train_dqn(env, agent, num_episodes=8000)