# Construction Risk Evaluation Tool (AI-Based)

## Overview

This project is an AI-powered tool that predicts construction project risk levels using machine-learning models such as Logistic Regression, Random Forest, and SMOTE-balanced variants.

## Project Structure

- `app.py` – Optional GUI / CLI interface
- `predict_risk.py` – Main script to run predictions
- `risk_model.pkl` – Trained model
- `scaler.pkl` – StandardScaler used in preprocessing
- `imputer.pkl` – Data imputer
- `model_classes.json` – Encoded class labels
- `construction_risk_dataset_2000.csv` – Dataset
- `feature_importances.png`, `confusion_matrix_*.png` – Evaluation plots
- `model_comparison.png` – Model accuracy comparison
- `requirements.txt` – Dependency list

## Setup Instructions

### Step 1: Clone the Repository

Download the project from GitHub:

```bash
git clone https://github.com/gzjz10/ConstructionRiskAI.git
cd ConstructionRiskAI
```

### Step 2: Install Requirements

If `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

Otherwise install manually:

```bash
pip install pandas scikit-learn matplotlib joblib
```

### Step 3: Run the Application

For command-line prediction:

```bash
python predict_risk.py
```

If you have a GUI interface:

```bash
python app.py
```

### Step 4: Review Output

The predicted risk category appears in the terminal or GUI. Evaluation plots (confusion matrices, feature importance, etc.) are saved to PNG files inside the project folder.

## Models Used

- ✅ **Logistic Regression** (best accuracy ≈ 98.75%)
- Random Forest
- Decision Tree
- RUSBoost
- SMOTE-balanced models

## Future Improvements

- Add Flask / FastAPI web interface
- Real-time data ingestion via APIs
- Deploy as a desktop or cloud application
- Experiment with deep-learning architectures
