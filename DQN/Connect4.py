import numpy as np
import torch

class Connect4:
    def __init__(self):
        """Initialize the game board and constants"""
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.rows= 6
        self.columns = 7
        self.board = self.create_board()
        self.game_over = False
        self.current_player = 1 # initializing with player 0's turn.
        self.debug = False

    def create_board(self):
        """Creates an empty game board (6 rows by 7 columns)"""
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))

    def drop_piece(self, row, col, piece):
        """Places the player's piece on the board"""
        self.board[row][col] = piece

    def is_valid_location(self, col):
        """Checks if the top of the selected column is still open"""
        return self.board[self.ROW_COUNT-1][col] == 0

    def get_next_open_row(self, col):
        """Finds the next open row in the given column"""
        for r in range(self.ROW_COUNT):
            if self.board[r][col] == 0:
                #print('PARI', r)
                #self.print_board()
                return r

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


    def is_winning_move(self, piece):
        """Checks if the current move resulted in a win"""
        # Check horizontal locations for win
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT):
                if (self.board[r][c] == piece and
                    self.board[r][c+1] == piece and
                    self.board[r][c+2] == piece and
                    self.board[r][c+3] == piece):
                    #print('HORIZONTAL')
                    return True

        # Check vertical locations for win
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT-3):
                if (self.board[r][c] == piece and
                    self.board[r+1][c] == piece and
                    self.board[r+2][c] == piece and
                    self.board[r+3][c] == piece):
                    #print('VERTICAL')
                    return True

        # Check positively sloped diagonals
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT-3):
                if (self.board[r][c] == piece and
                    self.board[r+1][c+1] == piece and
                    self.board[r+2][c+2] == piece and
                    self.board[r+3][c+3] == piece):
                    #print('POS DIAG')
                    return True

        # Check negatively sloped diagonals
        for c in range(self.COLUMN_COUNT-3):
            for r in range(3, self.ROW_COUNT):
                if (self.board[r][c] == piece and
                    self.board[r-1][c+1] == piece and
                    self.board[r-2][c+2] == piece and
                    self.board[r-3][c+3] == piece):
                    #print('NEG DIAG')
                    return True
    

    def is_draw(self):
    # Check if any spot on the board is empty (None or 0 represents an empty spot)
        for row in self.board:
            if 0 in row:  # Adjust this to match how you represent empty spaces
                return False
        return True
    
    def set_player_turn(self, player):
        if player == 0:
            self.current_player = -1
        else:
            self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = -1  # player turn is either -1 or 1
        return self.board
    
    def is_valid_action(self, action):
        return self.board[self.ROW_COUNT-1][action] == 0  # Check if the top row is empty
    
    def get_valid_actions(self):
        return [col for col in range(self.columns) if self.is_valid_action(col)]
    
    def step(self, action):
        # Action is the column where to drop the piece
        if self.is_valid_action(action):
            row = self.get_next_open_row(action)
            self.board[row][action] = self.current_player
            if self.debug:
                self.print_board()
            if self.is_winning_move(self.current_player):
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
            return self.board, -10, False  # Invalid move penalty


    def play(self):
        """Main game loop for playing Connect 4"""
        self.print_board()
        
        while not self.game_over:
            # Ask for Player 1 input
            if self.current_player == 1:
                col = int(input("Player 1, make your selection (0-6): "))
                piece = 1
            else:
                col = int(input("Player 2, make your selection (0-6): "))
                piece = -1

            if self.is_valid_location(col):
                row = self.get_next_open_row(col)
                self.drop_piece(row, col, piece)

                if self.is_winning_move(piece):
                    print(f"Player {piece} wins!")
                    self.game_over = True

            self.print_board()
            
            # Alternate between players
            if self.current_player == 1:
                self.current_player = -1
            else:
                self.current_player = 1

    
    def play_vs_dqn(self):
        piece = -1
        col = int(input("Player 2, make your selection (0-7): "))
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            self.drop_piece(row, col, piece)

            if self.is_winning_move(piece):
                print(f"Player {piece} wins!")
                self.game_over = True

        self.print_board()



            

# To play the game, create an instance of the class and call the play method:
if __name__ == "__main__":
    # game = Connect4()
    # game.play()
    pass
    
