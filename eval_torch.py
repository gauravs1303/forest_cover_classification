import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import os

# Import the model from your model.py file
from model_pytorch import ForestCoverNet

class ForestDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def run_evaluation(model_path, csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    # weights_only=False is used to allow loading of the full dictionary including numpy arrays
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Use input_dim from checkpoint if available, otherwise default to dataset standard (54)
    input_dim = checkpoint.get('input_dim', 54)
    
    # 2. Reconstruct Model
    model = ForestCoverNet(input_dim=input_dim, num_classes=7).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Load Data
    df = pd.read_csv(csv_path)
    X_raw = df.drop('Cover_Type', axis=1).values
    y_true = df['Cover_Type'].values - 1  # Convert 1-7 range to 0-6 index
    
    # 4. Preprocess using saved scaler stats
    # Dynamic detection of scaled features
    mean = checkpoint['scaler_mean']
    scale = checkpoint['scaler_scale']
    num_scaled = len(mean) 
    
    X_cont = (X_raw[:, :num_scaled] - mean) / scale
    X_final = np.concatenate([X_cont, X_raw[:, num_scaled:]], axis=1)
        
    loader = DataLoader(ForestDataset(X_final, y_true), batch_size=512, shuffle=False)
    
    # 5. Inference
    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    
    all_preds = np.array(all_preds)
    
    # 6. Calculate Metrics
    macro_f1 = f1_score(y_true, all_preds, average='macro')
    weighted_f1 = f1_score(y_true, all_preds, average='weighted')
    micro_f1 = f1_score(y_true, all_preds, average='micro')
    acc = accuracy_score(y_true, all_preds)
    cm = confusion_matrix(y_true, all_preds)
    report = classification_report(y_true, all_preds, digits=4, 
                                   target_names=["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", 
                                                "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"],
                                   output_dict=True)

    # ======================================================================
    # FORMATTED PRINTING
    # ======================================================================
    print("="*70)
    print("FOREST COVER TYPE CLASSIFICATION - MODEL EVALUATION (PYTORCH)")
    print("="*70)
    print(f"Device:           {device}")
    print(f"Test file:        {os.path.basename(csv_path)}")
    print(f"Model checkpoint: {os.path.basename(model_path)}")
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"{'OVERALL METRICS':^70}")
    print("-" * 70)
    print(f"{'Macro F1 Score (PRIMARY METRIC):':<46} {macro_f1:.6f}")
    print(f"{'Accuracy:':<46} {acc:.6f} ({acc*100:.2f}%)")
    print(f"{'Micro F1 Score:':<46} {micro_f1:.6f}")
    print(f"{'Weighted F1 Score:':<46} {weighted_f1:.6f}")
    print()
    print(f"{'PER-CLASS F1 SCORES':^70}")
    print("-" * 70)
    print(f"{'Class':<30} {'F1 Score':<15} {'Support':<15}")
    print("-" * 70)
    
    classes = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
    for cls in classes:
        f1 = report[cls]['f1-score']
        sup = int(report[cls]['support'])
        print(f"{cls:<30} {f1:<15.6f} {sup:<15}")

    print()
    print(f"{'DETAILED CLASSIFICATION REPORT':^70}")
    print("="*70)
    print(classification_report(y_true, all_preds, digits=4, target_names=classes))
    
    print(f"{'CONFUSION MATRIX':^70}")
    print("="*70)
    print("Rows: True Labels | Columns: Predicted Labels")
    print("-" * 70)
    header = "True\\Pred |" + "".join([f"   {i}   |" for i in range(7)])
    print(header)
    print("-" * 70)
    for i, row in enumerate(cm):
        row_str = f"Class {i}   |" + "".join([f" {val:<5} |" for val in row])
        print(row_str)
    
    print("="*70)
    print(f"🎯 FINAL MACRO F1 SCORE: {macro_f1:.6f}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='forest_cover_model.pth')
    args = parser.parse_args()
    
    run_evaluation(args.model_path, args.test_data)