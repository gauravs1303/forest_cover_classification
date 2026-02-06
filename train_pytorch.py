"""
train.py - Training Script for Forest Cover Classification

Students: Implement your complete training pipeline in this file.

REQUIRED OUTPUT:
Your script must save a model checkpoint named 'forest_cover_model.pth' containing:
- 'model_state_dict': model.state_dict()
- 'input_dim': number of input features
- 'scaler_mean': mean values used for standardization (numpy array)
- 'scaler_scale': scale values used for standardization (numpy array)

Example:
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, 'forest_cover_model.pth')
"""
import torch
import torch.nn as nn
from model_pytorch import ForestCoverNet
import pandas as pd
import numpy as np
import argparse
import os
import pickle

# TODO: Implement your complete training pipeline
# Start
def main():
    #1. Load data from 'covtype_train.csv'
    df = pd.read_csv(args.data_set)
    #2. Preprocess and split data
    X = df.iloc[:,:-1].values.astype(np.float32)
    Y = df.iloc[:, -1].values.astype(int) -1
    ## Standardise
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X = (X-mean)/(std+1e-6)
    # Numpy to torch tensor to avoid this eror, "TypeError: linear(): argument 'input' (position 1) must be Tensor, not numpy.ndarray"
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

    #3. Create model, define loss and optimizer
    ## Hyper-parameter
    lr = 0.01
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ForestCoverNet(input_dim=54)

    # Weighted class
    class_counts = torch.bincount(Y)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    loss_funct = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #4. Train the model
    ce = []
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            # Resetting the gradient to zero for this iteration
            optimizer.zero_grad()
            # Forward pass
            logits = model(xb)
            loss = loss_funct(logits, yb)
            # Backward Pass
            loss.backward()
            # Updating the parameters
            optimizer.step()
            epoch_loss +=loss.item()
            
        ce.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss/len(loader):.4f}")

    #5. Save model checkpoint with required keys
    torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': 54,
            'scaler_mean': mean,
            'scaler_scale': std
        }, 'CLASSIFICATION/forest_cover_model_pytorch.pth')
    print("Trainig is completer, Model is Saved")

    # Saving the logs
    logs = {
        "ce_loss" : ce
    }
    os.makedirs("CLASSIFICATION/logs", exist_ok=True)
    with open(f"CLASSIFICATION/logs/torch_hd{"-".join(map(str, model.hidden_dim))}_batch{batch_size}_ep{args.epochs}_lr{lr}.pkl", "wb") as f:
        pickle.dump(logs, f)

if __name__ == "__main__":
    # Your training code here
    parser = argparse.ArgumentParser(description="Train ForestCoverNet using Pytorch")
    parser.add_argument('--data_set', type=str, required=True, help='Path to training csv')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    main()