#Connect4 trained agent play

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
        
        return self.get_state(), 0, False, None
    
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


def play_against_agent(model_path, human_player=-1):
    # Initialize environment
    env = Connect4Env()
    action_space = env.action_space
    
    # Load the trained model
    model = DQN(action_space).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()  # Set to evaluation mode
    
    # Start game
    state = env.reset()
    done = False
    
    print("\n===== CONNECT 4 =====")
    print("You are 'O', AI is 'X'")
    print("Enter a column number (0-6) to make your move")
    
    # Choose who goes first
    first_player = input("Do you want to go first? (y/n): ").lower()
    if first_player == 'y':
        env.current_player = -1  # Human goes first
    
    while not done:
        env.render()
        
        if env.current_player == human_player:
            # Human's turn
            valid_move = False
            while not valid_move:
                try:
                    action = int(input("Your move (0-6): "))
                    if 0 <= action <= 6 and env._is_valid_move(action):
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
                q_values = model(state_tensor)
                
                # Mask invalid actions
                mask = torch.ones(action_space, device=device) * float('-inf')
                mask[valid_actions] = 0
                q_values = q_values + mask
                
                action = q_values.max(1)[1].item()
            print(f"AI chose column {action}")
        
        # Make move
        next_state, _, done, _ = env.step(action)
        state = next_state
        
        if done:
            env.render()
            if env.winner == 1:
                print("AI wins!")
            elif env.winner == -1:
                print("You win!")
            else:
                print("It's a draw!")
            
            # Ask to play again
            play_again = input("Play again? (y/n): ").lower()
            if play_again == 'y':
                state = env.reset()
                done = False
                first_player = input("Do you want to go first? (y/n): ").lower()
                if first_player == 'y':
                    env.current_player = -1  # Human goes first
                else:
                    env.current_player = 1  # AI goes first
            else:
                print("Thanks for playing!")


if __name__ == "__main__":
    model_path = input("Enter the path to your trained model (e.g., connect4_dqn_best.pth): ")
    play_against_agent(model_path)