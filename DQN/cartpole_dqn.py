import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

# Hyperparameters
BUFFER_SIZE = 10000     # Replay buffer size
BATCH_SIZE = 64         # Mini-batch size
GAMMA = 0.99            # Discount factor
LR = 0.001              # Learning rate
TARGET_UPDATE = 10      # Target network update frequency
EPSILON_START = 1.0     # Initial epsilon for exploration
EPSILON_END = 0.01      # Final epsilon
EPSILON_DECAY = 0.995   # Epsilon decay rate


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START

        # Q-network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        # Replay memory
        self.memory = deque(maxlen=BUFFER_SIZE)

    def remember(self, state, action, reward, next_state, done):
        if len(state)==4:
            self.memory.append((state, action, reward, next_state, done))
        else:
            print('Extradim', state)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        try:
            batch = random.sample(self.memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states)

            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()

        # Current Q values
        q_values = self.q_network(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# Training loop
def train_dqn(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            #print('DDDD', action, env.step(action))
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.replay()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")


import gym

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(state_size, action_size)
agent = DQNAgent(state_size, action_size)
train_dqn(env, agent, num_episodes=1000)