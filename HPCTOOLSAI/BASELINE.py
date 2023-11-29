import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile
import tqdm
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from torch.utils.tensorboard import SummaryWriter

# SummaryWriter for logging model results
writer = SummaryWriter("logs/BASELINE")

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Check if GPU is available
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    # If CUDA is not available, use the CPU
    print("GPU not available. Using CPU")

# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

# Loss function and optimizer
loss_fn = nn.MSELoss()  # Mean Square Error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 500  # Number of epochs to run
batch_size = 32  # Size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # Initialize to infinity
best_weights = None
history = []

# Initialize time
start_time = time.time()

# Executing without profiler
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # Take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            bar.set_postfix(mse=float(loss))
            batch_percent = 100.0 * (start.item() + batch_size) / len(X_train)
            bar.set_postfix(loss=loss.item(), batch_percent=batch_percent)

            # Log training loss to TensorBoard
            iteration = epoch * len(batch_start) + start.item()
            writer.add_scalar('Loss/train', loss.item(), iteration)
            
    # Evaluate accuracy at the end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)

    # Log RMSE to TensorBoard
    writer.add_scalar('RMSE/test', np.sqrt(mse), epoch)

    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# Close the SummaryWriter when done
writer.close()

# Total training time
end_time = time.time()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)

# Restore the model and return the best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
print(f"Total Training Time: {int(minutes)} minutes {int(seconds)} seconds")
plt.plot(history)
plt.show()
