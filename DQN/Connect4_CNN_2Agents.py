import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Constants
BOARD_ROWS = 6
BOARD_COLS = 7
ACTION_SIZE = BOARD_COLS
STATE_SHAPE = (1, BOARD_ROWS, BOARD_COLS)

class Connect4:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.current_player = 1
    
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        return self.board[np.newaxis, :, :]
    
    def is_valid_action(self, action):
        return self.board[0, action] == 0
    
    def step(self, action):
        if not self.is_valid_action(action):
            return self.get_state(), -10, True  # Invalid move penalty
        
        for row in range(BOARD_ROWS-1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        
        reward, done = self.check_winner()
        self.current_player = 3 - self.current_player  # Switch player
        return self.get_state(), reward, done
    
    def check_winner(self):
        for c in range(BOARD_COLS - 3):
            for r in range(BOARD_ROWS):
                if self.board[r, c] != 0 and all(self.board[r, c+i] == self.board[r, c] for i in range(4)):
                    return (1 if self.board[r, c] == 1 else -1), True
        
        for c in range(BOARD_COLS):
            for r in range(BOARD_ROWS - 3):
                if self.board[r, c] != 0 and all(self.board[r+i, c] == self.board[r, c] for i in range(4)):
                    return (1 if self.board[r, c] == 1 else -1), True
        
        for c in range(BOARD_COLS - 3):
            for r in range(BOARD_ROWS - 3):
                if self.board[r, c] != 0 and all(self.board[r+i, c+i] == self.board[r, c] for i in range(4)):
                    return (1 if self.board[r, c] == 1 else -1), True
        
        for c in range(BOARD_COLS - 3):
            for r in range(3, BOARD_ROWS):
                if self.board[r, c] != 0 and all(self.board[r-i, c+i] == self.board[r, c] for i in range(4)):
                    return (1 if self.board[r, c] == 1 else -1), True
        
        if np.all(self.board != 0):
            return 0, True  # Draw
        
        return 0, False

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * BOARD_ROWS * BOARD_COLS, 512)
        self.fc2 = nn.Linear(512, ACTION_SIZE)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
    
    def act(self, state, env):
        if random.random() < self.epsilon:
            return random.choice([a for a in range(ACTION_SIZE) if env.is_valid_action(a)])
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        valid_actions = [a for a in range(ACTION_SIZE) if env.is_valid_action(a)]
        return max(valid_actions, key=lambda a: q_values[0, a].item())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Training loop
env = Connect4()
agent1 = DQNAgent()
agent2 = DQNAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        current_agent = agent1 if env.current_player == 1 else agent2
        action = current_agent.act(state, env)
        next_state, reward, done = env.step(action)
        current_agent.remember(state, action, reward, next_state, done)
        state = next_state
    agent1.train()
    agent2.train()
    agent1.decay_epsilon()
    agent2.decay_epsilon()
    if episode % 50 == 0:
        agent1.update_target_model()
        agent2.update_target_model()
        print(f"Episode {episode}: Training in progress...")

print("Training complete!")
