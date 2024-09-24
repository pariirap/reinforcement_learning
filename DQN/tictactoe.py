import numpy as np
import torch

import numpy as np

class TicTacToe:
    def __init__(self):
        # Initialize a 3x3 board with empty strings
        self.board = np.full((3, 3), 0)
        self.current_player = 1  # X always goes first
        self.debug= False
    
    def set_player_turn(self, player):
        if player == 1:
            self.current_player = 1
        else:
            self.current_player = -1
    
    def print_board(self):
        # Print the board in a readable format
        for row in self.board:
            print(' | '.join([str(cell) if cell != 0 else ' ' for cell in row]))
            print('-' * 9)

    def get_board_state(self):
        # Convert NumPy grid to PyTorch tensor
        tensor_grid = torch.from_numpy(self.board).float()
        flattened_grid = tensor_grid.view(-1) 
        return flattened_grid
    
    def make_move(self, row, col):
        # Make a move at the specified position
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if self.is_winning_move():
                print(f"Player {self.current_player} wins!")
                self.print_board()
                return True
            elif self.is_draw():
                print("It's a draw!")
                self.print_board()
                return True
            self.current_player = -1 if self.current_player == 1 else 1
        else:
            print("Invalid move, try again.")
        return False
    
    def is_winning_move(self):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player) or np.all(self.board[:, i] == self.current_player):
                return True
        if np.all(np.diag(self.board) == self.current_player) or np.all(np.diag(np.fliplr(self.board)) == self.current_player):
            return True
        return False
    
    def is_draw(self):
        # Check if all cells are filled and no winner
        return np.all(self.board != 0)
    
    
    def index_to_row_col(self, index):
        if 0 <= index <= 8:
            row = index // 3  # Get the row by dividing the index by 3
            col = index % 3   # Get the column by the remainder when dividing by 3
            return row, col
        else:
            raise ValueError("Index must be between 0 and 8 (inclusive).") 
    
    def reset(self):
        # Reset the board for a new game
        self.board = np.full((3, 3), 0)
        self.current_player = 1

    def is_valid_action(self, action): # make sure you are not over stepping
        r, c = self.index_to_row_col(action)
        return self.board[r][c] == 0


    def step(self, action):
        # Action is the column where to drop the piece
        if self.is_valid_action(action):
            row, col = self.index_to_row_col(action)
            self.board[row][col] = self.current_player
            if self.debug:
                self.print_board()
            if self.is_winning_move():
                reward = 1  # Reward for winning
                done = True
                #print('WINNER', self.current_player)
                #self.print_board()
            elif self.is_draw():
                reward = 0  # Draw
                print('*******************DRAW********************************************************')
                done = True
            else:
                reward = 0  # Continue the game
                done = False
                self.current_player *= -1  # Switch player
            return self.board, reward, done
        else:
            return self.board, -1, True  # Invalid move penalty
        

# Example usage
game = TicTacToe()
game.print_board()
game.make_move(0, 0)
game.make_move(0, 2)
game.make_move(1, 1)
game.make_move(1, 0)
game.make_move(2, 2)  # X wins

    