# Classification –  Forest Cover Type Dataset

This folder contains the **classification task** implementation using:
- Pure **NumPy**
- **PyTorch**
- Detailed comparison of loss curves and training behavior

The code structure follows the assignment PDF and uses reference files only as a starting point.

---

## 📂 Folder Structure
> **Note:** You might have to change the Observation.py file according to the folder refernce and desired outputs.

```text
CLASSIFICATION/
│
├── logs/ # Pickle logs for NumPy & PyTorch training
├── plots/ # Loss comparison plots
│
├── covtype_train.csv # Training dataset
│
├── train_numpy.py # Neural network classifier (NumPy)
├── train_pytorch.py # Neural network classifier (PyTorch)
│
├── model_numpy.py # NumPy model definition
├── model_pytorch.py # PyTorch model definition
│
├── eval_numpy.py # Evaluation for NumPy model
├── eval_torch.py # Evaluation for PyTorch model
│
├── forest_cover_model # Saved trained models
├── Observation.py # Comparison & visualization script
```

---

## 📌 Reference Usage

- Dataset loading and structure were inspired by the provided reference
- Model architectures, training loops, loss tracking, and comparisons were fully implemented and customized


## Dataset Overview
- **Target Variable:** Cover Type (Integer values 1–7)  
- **Forest Cover Types:**
  1. Spruce/Fir  
  2. Lodgepole Pine  
  3. Ponderosa Pine  
  4. Cottonwood/Willow  
  5. Aspen  
  6. Douglas-fir  
  7. Krummholz
- **Features:**
  - Elevation  
  - Aspect  
  - Slope  
  - Horizontal_Distance_To_Hydrology  
  - Vertical_Distance_To_Hydrology  
  - Horizontal_Distance_To_Roadways  
  - Hillshade_9am  
  - Hillshade_Noon  
  - Hillshade_3pm  
  - Horizontal_Distance_To_Fire_Points  
  - Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4  
  - Soil_Type1 through Soil_Type40  

---

## 🚀 How to Run

### 1️⃣ NumPy Classifier
- **Train**

```bash
python train_numpy.py \
    --data_set covtype_train.csv \
    --epochs 30

Logs saved in: logs/
```
- **Evaluation**
```bash
python eval_numpy.py \
    --test_data covtype_train.csv \
    --model_path forest_cover_model_numpy.pth

```

### 2️⃣ PyTorch Classifier
- **Train**

```bash
python train_numpy.py \
    --data_set covtype_train.csv \
    --epochs 30

Logs saved in: logs/
```
- **Evaluation**
```bash
python eval_numpy.py \
    --test_data covtype_train.csv \
    --model_path forest_cover_model_numpy.pth

```

3️⃣ Run Observations & Comparisons
```text
python Observation.py
```

This script:

- Compares NumPy vs PyTorch with identical hyperparameters

- Compares different hyperparameter settings

- Plots Cross-Entropy Loss vs Epochs

Plots saved in: plots/

---

## 🧪 Observations
- PyTorch converges faster and is numerically more stable

- NumPy implementation offers full transparency but requires careful tuning

- Loss curves validate correctness across both frameworks

---

## 🛠 Notes
- PyTorch implementation can utilize GPU (if available)

- NumPy implementation is CPU-only

- Logs are stored as .pkl files for reproducibility