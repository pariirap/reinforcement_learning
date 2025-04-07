# We use CNN with 3 input channels. Two channels for each opponent and the third for valid moves. 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Connect4 Environment
class Connect4Env:
    def __init__(self):
        self.reset()
        self.action_space = 7  # 7 columns in Connect4
        
    def reset(self):
        # 0 for empty, 1 for player 1 (agent), -1 for player 2 (opponent)
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.winner = None
        return self.get_state()
    
    def get_state(self):
        # Convert the board to a 3-channel representation for the CNN
        # Channel 1: Agent's pieces (1s where agent has pieces, 0s elsewhere)
        # Channel 2: Opponent's pieces (1s where opponent has pieces, 0s elsewhere)
        # Channel 3: Valid moves (1s in columns that aren't full)
        state = np.zeros((3, 6, 7), dtype=np.float32)
        state[0] = (self.board == 1).astype(np.float32)
        state[1] = (self.board == -1).astype(np.float32)
        
        # Valid moves (columns that aren't full)
        for col in range(7):
            if self._is_valid_move(col):
                # Mark the topmost empty position in this column
                for row in range(6):
                    if self.board[row, col] == 0:
                        state[2, row, col] = 1
                        break
        
        return state
    
    def _is_valid_move(self, col):
        return col >= 0 and col < 7 and self.board[5, col] == 0
    
    def _get_next_open_row(self, col):
        for row in range(6):
            if self.board[row, col] == 0:
                return row
        return -1
    
    def _check_win(self, player):
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if (self.board[row, col] == player and 
                    self.board[row, col+1] == player and 
                    self.board[row, col+2] == player and 
                    self.board[row, col+3] == player):
                    return True
        
        # Check vertical
        for row in range(3):
            for col in range(7):
                if (self.board[row, col] == player and 
                    self.board[row+1, col] == player and 
                    self.board[row+2, col] == player and 
                    self.board[row+3, col] == player):
                    return True
        
        # Check diagonal (positive slope)
        for row in range(3):
            for col in range(4):
                if (self.board[row, col] == player and 
                    self.board[row+1, col+1] == player and 
                    self.board[row+2, col+2] == player and 
                    self.board[row+3, col+3] == player):
                    return True
        
        # Check diagonal (negative slope)
        for row in range(3, 6):
            for col in range(4):
                if (self.board[row, col] == player and 
                    self.board[row-1, col+1] == player and 
                    self.board[row-2, col+2] == player and 
                    self.board[row-3, col+3] == player):
                    return True
        
        return False
    
    def _is_board_full(self):
        return not any(self.board[5, col] == 0 for col in range(7))
    
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, None
        
        if not self._is_valid_move(action):
            return self.get_state(), -10, True, {"invalid_move": True}  # Penalty for invalid move
        
        # Make the move
        row = self._get_next_open_row(action)
        self.board[row, action] = self.current_player
        
        # Check if the current player won
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1 if self.current_player == 1 else -1
            return self.get_state(), reward, True, {"winner": self.current_player}
        
        # Check if the board is full (draw)
        if self._is_board_full():
            self.done = True
            self.winner = 0
            return self.get_state(), 0.1, True, {"draw": True}  # Small positive reward for draw
        
        # Switch player
        self.current_player *= -1
        
        # If it's opponent's turn, let them make a random valid move
        if self.current_player == -1:
            valid_cols = [col for col in range(7) if self._is_valid_move(col)]
            if valid_cols:
                opponent_action = random.choice(valid_cols)
                return self.step(opponent_action)
        
        # Small positive reward for not losing yet
        return self.get_state(), 0.01, False, None
    
    def render(self):
        print("\n")
        for row in range(5, -1, -1):
            line = "|"
            for col in range(7):
                if self.board[row, col] == 1:
                    line += "X|"
                elif self.board[row, col] == -1:
                    line += "O|"
                else:
                    line += " |"
            print(line)
        print("---------------")
        print("|0|1|2|3|4|5|6|")
        print("\n")


# CNN-based DQN Model
class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, action_space)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Experience Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(
        self,
        action_space,
        replay_buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        lr=0.001,
        target_update_freq=10
    ):
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        # Q-networks
        self.policy_net = DQN(action_space).to(device)
        self.target_net = DQN(action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(replay_buffer_size)
        
    def select_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.action_space))
            
        # Epsilon-greedy action selection
        if random.random() < self.eps:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                
                # Mask invalid actions with a large negative value
                mask = torch.ones(self.action_space) * float('-inf')
                mask[valid_actions] = 0
                q_values = q_values + mask.to(device)
                
                return q_values.max(1)[1].item()
    
    def update_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(device)
        
        # Handle terminal states
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: not s, batch.done))).to(device)
        non_final_next_states = torch.FloatTensor([s for s, d in zip(batch.next_state, batch.done) if not d]).to(device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1})
        next_state_values = torch.zeros(self.batch_size, 1, device=device)
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1, keepdim=True)[0]
        
        # Compute expected Q values
        expected_q_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()


# Training function
def train_agent(
    agent,
    env,
    num_episodes=2000,
    max_steps=100,
    eval_freq=100,
    eval_episodes=10,
    save_path="connect4_dqn.pth"
):
    rewards = []
    losses = []
    eval_rewards = []
    best_eval_reward = float('-inf')
    
    print("Starting training...")
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        
        for step in range(max_steps):
            # Get valid actions
            valid_actions = [col for col in range(7) if env._is_valid_move(col)]
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.memory.push(state, action, next_state, reward, done)
            
            # Learn
            loss = agent.learn()
            if loss:
                episode_loss += loss
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # Update epsilon
        agent.update_epsilon()
        
        # Track metrics
        rewards.append(episode_reward)
        if episode_loss > 0:
            losses.append(episode_loss / (step + 1))
        else:
            losses.append(0)
            
        # Evaluate agent periodically
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {np.mean(rewards[-eval_freq:]):.3f}, Eval Reward: {eval_reward:.3f}, Epsilon: {agent.eps:.3f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(agent.policy_net.state_dict(), save_path)
                print(f"New best model saved with eval reward: {eval_reward:.3f}")
    
    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(range(eval_freq-1, num_episodes, eval_freq), eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    return rewards, losses, eval_rewards


# Evaluation function
def evaluate_agent(agent, env, num_episodes=10, render=False):
    total_rewards = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get valid actions
            valid_actions = [col for col in range(7) if env._is_valid_move(col)]
            
            # Select best action (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor)
                
                # Mask invalid actions
                mask = torch.ones(agent.action_space) * float('-inf')
                mask[valid_actions] = 0
                q_values = q_values + mask.to(device)
                
                action = q_values.max(1)[1].item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            state = next_state
        
        total_rewards += episode_reward
    
    return total_rewards / num_episodes


# Function to play against the trained agent
def play_against_agent(agent, env, human_player=-1):
    state = env.reset()
    done = False
    
    print("Connect 4 Game")
    print("You are 'O', AI is 'X'")
    print("Enter a column number (0-6) to make your move")
    
    while not done:
        env.render()
        
        if env.current_player == human_player:
            # Human's turn
            valid_move = False
            while not valid_move:
                try:
                    action = int(input("Your move (0-6): "))
                    if action >= 0 and action <= 6 and env._is_valid_move(action):
                        valid_move = True
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 6.")
        else:
            # Agent's turn
            print("AI is thinking...")
            valid_actions = [col for col in range(7) if env._is_valid_move(col)]
            
            # Select best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor)
                
                # Mask invalid actions
                mask = torch.ones(agent.action_space) * float('-inf')
                mask[valid_actions] = 0
                q_values = q_values + mask.to(device)
                
                action = q_values.max(1)[1].item()
            print(f"AI chose column {action}")
        
        # Make move
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        if done:
            env.render()
            if env.winner == 1:
                print("AI wins!")
            elif env.winner == -1:
                print("You win!")
            else:
                print("It's a draw!")


# Main function to run the entire training and evaluation process
def main():
    # Initialize environment and agent
    env = Connect4Env()
    agent = DQNAgent(
        action_space=env.action_space,
        replay_buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        lr=0.001,
        target_update_freq=10
    )
    
    # Train the agent
    rewards, losses, eval_rewards = train_agent(
        agent=agent,
        env=env,
        num_episodes=2000,  # Increase for better performance
        max_steps=100,
        eval_freq=100,
        eval_episodes=10,
        save_path="connect4_dqn_best.pth"
    )
    
    # Load the best model
    agent.policy_net.load_state_dict(torch.load("connect4_dqn_best.pth"))
    
    # Evaluate the trained agent
    print("\nFinal Evaluation:")
    eval_reward = evaluate_agent(agent, env, num_episodes=20, render=False)
    print(f"Average Reward over 20 episodes: {eval_reward:.3f}")
    
    # Play against the agent
    play_option = input("Do you want to play against the AI? (y/n): ")
    if play_option.lower() == 'y':
        play_against_agent(agent, env)


if __name__ == "__main__":
    main()