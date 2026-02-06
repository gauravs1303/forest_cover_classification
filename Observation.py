import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "CLASSIFICATION/logs"
SAVE_DIR = "CLASSIFICATION/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


file = os.listdir(LOG_DIR)
log_file = {"numpy": [x for x in file if "numpy" in x],
            "torch": [x for x in file if "torch" in x]}

# Function for comparison between numpy and torch with same hyper parameter
def numpyvtorch(numpy_list, torch_list, if_plot=False):
    for n in numpy_list:
        for t in torch_list:
            if n.replace("numpy", "") == t.replace("torch", ""):
                plt.figure(figsize=(10, 6))
                plt.title = f"Numpy Vs Torch with hyper parameters : {n.replace("numpy", "").replace(".pkl","")}"

                for log_file in (t,n):
                    log_path = os.path.join(LOG_DIR, log_file)

                    with open(log_path, "rb") as f:
                        logs = pickle.load(f)

                    if "ce_loss" not in logs:
                        continue

                    ce = logs["ce_loss"]
                    epochs = np.arange(len(ce))

                    # Detect framework
                    if "numpy" in log_file.lower():
                        label = f"NumPy ({log_file.replace("_", " ").replace(".pkl", "")})"
                        linestyle = "--"
                    elif "torch" in log_file.lower():
                        label = f"PyTorch ({log_file.replace("_", " ").replace(".pkl", "")})"
                        linestyle = "-"
                    else:
                        label = log_file
                        linestyle = ":"

                    plt.plot(epochs, ce, linestyle=linestyle, label=label)

                if if_plot:
                    # Plot formatting
                    plt.xlabel("Epochs")
                    plt.ylabel("Cross-Entropy Loss")
                    plt.legend()
                    plt.grid(True)

                    # Save and show
                    plt.tight_layout()
                    plt.savefig(os.path.join(SAVE_DIR, f"Numpy_V_Torch{n.replace("numpy", "").replace(".pkl","")}.png"))

    

# Function for comparision between hyper parameters
def comparehyper(file_list, framework, if_plot=False):
    plt.figure(figsize=(10, 6))
    plt.title = f"{framework} Hyper parameter Plot"

    for log_file in file_list:
        log_path = os.path.join(LOG_DIR, log_file)

        with open(log_path, "rb") as f:
            logs = pickle.load(f)

        if "ce_loss" not in logs:
            continue

        ce = logs["ce_loss"]
        epochs = np.arange(len(ce))

        # Detect framework
        if "numpy" in log_file.lower():
            label = f"NumPy ({log_file.replace("_", " ").replace(".pkl", "")})"
            linestyle = "--"
        elif "torch" in log_file.lower():
            label = f"PyTorch ({log_file.replace("_", " ").replace(".pkl", "")})"
            linestyle = "-"
        else:
            label = log_file
            linestyle = ":"
        print(f"{log_file.replace("_", " | ").replace("-", ",").replace("hd", "HL ").replace("ep", "epoch ").replace("lr", "lr ").replace(".pkl", "")}")
        print(f"Final Cross Entropy Loss : {ce[-1]}")
        plt.plot(epochs, ce, linestyle=linestyle, label=label)

    if if_plot:
        # Plot formatting
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.grid(True)

        # Save and show
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{framework}_hypeparameter_comparision.png"))

# Compare hyper parameter
for key, values in log_file.items():
    comparehyper(values, key)

# Compare Numpy and torch
numpyvtorch(log_file["numpy"], log_file["torch"])