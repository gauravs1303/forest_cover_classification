import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
# Ensure model_numpy.py defines ForestCoverNet appropriately for NumPy inference
from model_numpy import ForestCoverNet

def print_formatted_report(y_true, y_pred, test_file, model_path):
    classes = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    output = []
    output.append("="*70)
    output.append("FOREST COVER TYPE CLASSIFICATION - MODEL EVALUATION (NUMPY)")
    output.append("="*70)
    output.append(f"Device: CPU (Pure NumPy)")
    output.append(f"Test file: {test_file}")
    output.append(f"Model checkpoint: {model_path}")
    output.append("="*70)
    output.append("EVALUATION RESULTS")
    output.append("="*70)
    output.append(f"{'OVERALL METRICS':^70}")
    output.append("-" * 70)
    output.append(f"Macro F1 Score (PRIMARY METRIC):              {macro_f1:.6f}")
    output.append(f"Accuracy:                                     {acc:.6f} ({acc*100:.2f}%)")
    output.append(f"Micro F1 Score:                               {micro_f1:.6f}")
    output.append(f"Weighted F1 Score:                            {weighted_f1:.6f}")
    output.append("\n" + f"{'PER-CLASS F1 SCORES':^70}")
    output.append("-" * 70)
    output.append(f"{'Class':<30} {'F1 Score':<15} {'Support':<15}")
    output.append("-" * 70)
    
    cls_f1s = f1_score(y_true, y_pred, average=None)
    # Ensure support count matches the number of classes in the report
    counts = np.bincount(y_true, minlength=7)
    for i, name in enumerate(classes):
        output.append(f"{name:<30} {cls_f1s[i]:<15.6f} {counts[i]:<15}")

    output.append("\n" + f"{'DETAILED CLASSIFICATION REPORT':^70}")
    output.append("="*70)
    output.append(report)
    output.append("\n" + f"{'CONFUSION MATRIX':^70}")
    output.append("="*70)
    output.append("Rows: True Labels | Columns: Predicted Labels")
    output.append("-" * 70)
    header = "True\\Pred |" + "".join([f"{i:^7}|" for i in range(7)])
    output.append(header)
    output.append("-" * 70)
    for i, row in enumerate(cm):
        row_str = f"Class {i}   |" + "".join([f"{val:^7}|" for val in row])
        output.append(row_str)
    output.append("="*70)
    output.append(f"🎯 FINAL MACRO F1 SCORE: {macro_f1:.6f}")
    output.append("="*70)

    final_text = "\n".join(output)
    print(final_text)
    with open('evaluation_results_numpy.txt', 'w', encoding='utf-8') as f:
        f.write(final_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    # 1. Load Data
    df = pd.read_csv(args.test_data)
    X_raw = df.drop('Cover_Type', axis=1).values
    y = df['Cover_Type'].values - 1

    # 2. Load Checkpoint
    with open(args.model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # 3. Dynamic Preprocessing (Consistency with eval_torch.py)
    # Retrieve scaling parameters from checkpoint
    if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
        mean = checkpoint['scaler_mean']
        scale = checkpoint['scaler_scale']
        
        # Determine how many features were scaled based on length of mean array
        num_scaled = len(mean)
        
        # Apply standardization to the chosen subset of features
        X_cont = (X_raw[:, :num_scaled] - mean) / scale
        X_final = np.concatenate([X_cont, X_raw[:, num_scaled:]], axis=1)
    else:
        # Fallback if no scaler info is found
        X_final = X_raw

    # 4. Reconstruct Model
    # input_dim should match the total features (54 for Forest Cover)
    input_dim = checkpoint.get('input_dim', X_final.shape[1])
    model = ForestCoverNet(input_dim=input_dim, num_classes=7)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    # 5. Inference
    preds = []
    batch_size = 512
    for i in range(0, len(X_final), batch_size):
        batch = X_final[i : i + batch_size]
        out = model.forward(batch)
        preds.extend(np.argmax(out, axis=1))

    print_formatted_report(y, np.array(preds), args.test_data, args.model_path)

if __name__ == "__main__":
    main()