import numpy as np
import torch

# Example: 2D NumPy grid of size 8x8
numpy_grid = np.random.rand(8, 8)

# Convert NumPy grid to PyTorch tensor
tensor_grid = torch.from_numpy(numpy_grid).float()

# If you're using a batch, add an extra dimension (batch_size, 8, 8)
tensor_grid = tensor_grid.unsqueeze(0)  # (1, 8, 8)

# Flatten the grid to feed it into a fully connected network
flattened_grid = tensor_grid.view(-1)  # Shape becomes (64,)


import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First layer
        self.fc2 = nn.Linear(128, 64)         # Second layer
        self.fc3 = nn.Linear(64, output_size) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the network (input size is 8*8 = 64, output size is user-defined)
input_size = 8 * 8 + 8  # The grid size when flattened
output_size = 10    # Example output size (you can adjust based on your task)
model = FCN(input_size, output_size)

# Forward pass through the network
print(flattened_grid, type(flattened_grid), flattened_grid.shape)
new_element = torch.tensor([-1]*8)
flattened_grid_1 = torch.cat((flattened_grid, new_element))
print(flattened_grid_1.shape)
print('I', flattened_grid_1)
output = model(flattened_grid_1)
print('O', output)