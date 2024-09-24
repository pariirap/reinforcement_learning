import numpy as np
import torch

class TweetyBird:
    def __init__(self):
        """Initialize the game board and constants"""
        self.rows= 3
        self.cols = 3
        self.board = self.create_board()
        self.game_over = False
        self.current_player = 1 # initializing with player 0's turn.
        self.debug = False

    def create_board(self):
        """Creates an empty game board (6 rows by 7 columns)"""
        return np.zeros((self.rows, self.cols))


    def is_valid_action(self, row, col):
        """Checks if the top of the selected column is still open"""
        if col == 0:
            return row >= 0
        if col == 1:
            return row == 2
        if col == 2:
            return row == 1
        return False


    def set_debug(self, debug):
        self.debug = debug

    def print_board(self):
        """Prints the board (flipped for user-friendly display)"""
        print(np.flip(self.board, 0))

    def get_board_state(self):
        # Convert NumPy grid to PyTorch tensor
        tensor_grid = torch.from_numpy(self.board).float()
        flattened_grid = tensor_grid.view(-1) 
        return flattened_grid


    def is_winning_move(self):
        """Checks if the current move resulted in a win"""
        # Check b[1][2] == 1 for win
        return self.board[self.rows-2][self.cols-1] == 1
        
       
    
    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        
        return self.board
    

    def step(self, action, col): # here action = row
        # Action is the column where to drop the piece
        if self.is_valid_action(action, col):
            self.board[action][col] = self.current_player
            if self.debug:
                self.print_board()
            if self.is_winning_move():
                reward = 1  # Reward for winning
                done = True
            else:
                self.board[action][col] = 1 
                reward = 0
                done = False
            return self.board, reward, done
        else:
            return self.board, -10, True  # Invalid move penalty


    def play(self):
        """Main game loop for playing Connect 4"""
        self.print_board()
        count = 0 
        done = False
        while not done and count < self.cols:
            
            row = int(input("Player 1, make your selection (0-6): "))
            b, r, d = self.step(row, count)
            done = d
            print(r, d)
            self.print_board()
            count += 1
        
            

# To play the game, create an instance of the class and call the play method:
if __name__ == "__main__":
    # game = TweetyBird()
    # game.play()
    pass