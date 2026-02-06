import numpy as np
import pandas as pd
import argparse
import pickle
from model_numpy import ForestCoverNet
import os

"""
REQUIRED OUTPUT:
Your script must save a model checkpoint named 'forest_cover_model_numpy.pth' containing:
- 'model_state_dict': dictionary of numpy arrays (weights/biases)
- 'input_dim': number of input features
- 'scaler_mean': mean values used for standardization
- 'scaler_scale': scale values used for standardization
"""

# Softmax
def softmax(z):
    # For numeric stability 
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

# Cross Entropy loss
def cross_entropy(Y_true, Y_pred):
    # Clipping to avoid log(0)
    eps = 1e-12
    Y_pred = np.clip(Y_pred, eps, 1. - eps)
    ce = -np.sum(Y_true*np.log(Y_pred)) / Y_true.shape[0]
    return ce

def one_hot(y, num_classes=7):
    return np.eye(num_classes)[y]

def train(args):
    # TODO: Implement complete training pipeline

    # Start
    # 1. Load data from args.data_set
    df = pd.read_csv(args.data_set)
    X = df.iloc[:, :-1].values.astype(np.float32)
    Y = df.iloc[:,-1].values.astype(int) - 1

    # 2. Standardise
    ## Mean and Std of X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Standardising X
    X = (X-mean)/ (std + 1e-6)
    ## One hot label for Y
    Y_oh = one_hot(Y)

    # 3. Initialize ForestCoverNet(input_dim=54)
    model = ForestCoverNet(input_dim=54)
    # Setting Hyper parameters
    lr = 0.01
    batch_size = 64
    # Training loop
    ce = []
    for epoch in range(args.epochs):
        # Shuffling data
        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        Y_oh = Y_oh[perm]

        epoch_loss =0.0
        for i in range(0, X.shape[0], batch_size):
            Xb = X[i:i+batch_size]
            Yb = Y_oh[i:i+batch_size]
            ## Forward
            logits = model.forward(Xb)
            probs = softmax(logits)
            loss = cross_entropy(Yb, probs)
            epoch_loss +=loss

            ## Backward
            delta = (probs-Yb)/Xb.shape[0]

            grad_W = {}
            grad_B = {}

            ### Output layer
            grad_W[model.L] = model.A[model.L].T @ delta
            grad_B[model.L] = np.sum(delta, axis=0, keepdims=True)

            ### Hidden Layers
            for l in range(model.L-1,-1, -1):
                # Updating delta
                delta = delta @ model.W[l+1].T
                # Relu gradient
                delta[model.Z[l]<=0]=0
                # Weight and bias gradient
                grad_W[l] = model.A[l].T @ delta
                grad_B[l] = np.sum(delta, axis=0, keepdims=True)
                    
            
            # SGD update parameters
            for l in model.W:

                model.W[l] -= lr*grad_W[l]
                model.B[l] -= lr*grad_B[l]
        epoch_loss_av = epoch_loss / (X.shape[0] // batch_size)
        ce.append(epoch_loss_av)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss_av:.2f}")

    # 4. Save the dictionary using pickle
    data = {
        "model_state_dict": {"W": model.W, "B":model.B},
        "input_dim": X.shape[1],
        "scaler_mean": mean,
        "scaler_scale": std
    }

    with open("CLASSIFICATION/forest_cover_model_numpy.pth", "wb") as f:
        pickle.dump(data, f)
    print("Model is trained and saved")

    #5. Saving logs
    logs = {
        "ce_loss" : ce
    }
    os.makedirs("CLASSIFICATION/logs", exist_ok=True)
    with open(f"CLASSIFICATION/logs/numpy_hd{"-".join(map(str,model.hidden_dim))}_batch{batch_size}_ep{args.epochs}_lr{lr}.pkl", "wb") as f:
        pickle.dump(logs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ForestCoverNet using NumPy")
    parser.add_argument('--data_set', type=str, required=True, help='Path to training csv')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    train(args)