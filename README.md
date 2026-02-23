# Neural Network — Pima Indians Diabetes

## Problem Statement
Binary classification task to predict whether a patient has diabetes based on 8 clinical measurements. Data sourced from the Pima Indians Diabetes dataset hosted on the UCI Machine Learning Repository.

## Dataset
| Property | Details |
|---|---|
| Source | Kaggle — [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| File | `diabetes.csv` (download from Kaggle and place in same folder as notebook) |
| Samples | 768 patients |
| Features | 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) |
| Target | `Outcome` — 0 = No Diabetes, 1 = Diabetes |
| Class balance | ~65% No Diabetes / ~35% Diabetes |

## Approach
The neural network pipeline follows the same workflow established in the Heart Disease lab:

1. **Load** data directly from the UCI URL using `pandas.read_csv`
2. **Explore** class distribution and feature ranges
3. **Preprocess**: 80/20 stratified train/test split + StandardScaler normalization
4. **Build** a fully connected neural network: `64 → Dropout(0.3) → 32 → Dropout(0.2) → 16 → 1`
5. **Train** with Adam optimizer, binary cross-entropy loss, and early stopping (patience=10)
6. **Evaluate** accuracy, precision, recall, F1-score, and confusion matrix

## Architecture
```
Input (8 features)
  └─ Dense(64, relu)
  └─ Dropout(0.3)
  └─ Dense(32, relu)
  └─ Dropout(0.2)
  └─ Dense(16, relu)
  └─ Dense(1, sigmoid)   ← output probability
```

## Results
| Metric | Score |
|---|---|
| Test Accuracy | ~74–76% |
| Precision | ~0.68–0.72 |
| Recall | ~0.62–0.68 |
| F1-Score | ~0.65–0.70 |

> Results are within the expected 70–75% range. Exact values depend on hardware randomness at runtime.

## Key Findings
- **Dropout regularization** was critical — without it, validation accuracy fell well below training accuracy due to the small dataset size
- **Recall** is the most important metric clinically, as missing a diabetes diagnosis has serious health consequences
- The Pima dataset is harder than Heart Disease because several features (insulin, skin thickness) contain zero values that likely represent missing data, adding noise
- Lowering the classification threshold from 0.5 to ~0.4 can improve recall at a slight cost to precision — worth considering for a screening context

## Repository Structure
```
neural-network-diabetes/
├── NN.ipynb
├── diabetes.csv
├── pima_diabetes_nn.ipynb    # Complete notebook (load → explore → train → evaluate)
├── README.md                  # This file
└── results/
    ├── training_curves.png    # Accuracy & loss over epochs
    ├── confusion_matrix.png   # TP/TN/FP/FN visualization
    └── metrics_summary.txt    # All metrics in plain text
```

## How to Run
```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn
jupyter notebook pima_diabetes_nn.ipynb
```

## How to Run
1. Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. Place `diabetes.csv` in the same folder as the notebook
3. Install dependencies and run:
```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn
jupyter notebook pima_diabetes_nn.ipynb
```
> **Colab users**: Uncomment the `files.upload()` lines at the top of the loading cell to upload the CSV directly.
