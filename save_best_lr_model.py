# save_best_lr_model.py
import pandas as pd
import numpy as np
from collections import Counter
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE # For SMOTE

import warnings
warnings.filterwarnings('ignore')

# --- Configuration from your training script ---
numeric_features = [
    'structural_complexity_rating', 'temporary_structures_rating', 'bim_adoption_percentage',
    'lead_time_rating', 'tariff_rate_percentage', 'supply_chain_risk_percentage',
    'socio_political_risk_score', 'permit_approval_rating', 'lien_claims_rating',
    'seismic_zone_rating', 'flood_risk_percentage', 'rainfall_flood_risk_percentage',
    'contingency_budget_percentage', 'weather_delay_rating', 'safety_risk_rating',
    'worker_experience_rating', 'turnover_rating', 'cybersecurity_risk_rating',
    'renewable_energy_percentage', 'energy_efficiency_percentage'
]
categorical_features = ['project_type', 'project_size', 'region']
exclude_features = [
    'project_id', 'project_name', 'start_date', 'planned_end_date',
    'delay_days', 'cost_overrun_percentage', 'safety_incidents',
    'overall_risk_score', 'actual_end_date', 'on_time_completion',
    'within_budget', 'project_success', 'high_risk_factors_count',
    'top_risk_factors', 'recommended_mitigations'
]
# --- End Configuration ---

print("Loading construction risk dataset (construction_risk_dataset_2000.csv)...")
try:
    df = pd.read_csv('construction_risk_dataset_2000.csv')
except FileNotFoundError:
    print("ERROR: 'construction_risk_dataset_2000.csv' not found.")
    print("Please ensure this CSV file is in the same directory.")
    exit()

print(f"Dataset shape: {df.shape}")
print("\nOriginal class distribution in CSV ('risk_category' column):")
if 'risk_category' in df.columns:
    print(df['risk_category'].value_counts())
else:
    print("ERROR: 'risk_category' column not found in CSV.")
    exit()

# Feature selection
X = df.drop(exclude_features + ['risk_category'], axis=1, errors='ignore')
y = df['risk_category']

# Ensure all defined numeric and categorical features are present in X
print("\nChecking for feature columns in the loaded DataFrame...")
actual_numeric_features = [f for f in numeric_features if f in X.columns]
actual_categorical_features = [f for f in categorical_features if f in X.columns]

if len(actual_numeric_features) != len(numeric_features):
    missing = set(numeric_features) - set(actual_numeric_features)
    print(f"Warning: Missing numeric features from CSV: {missing}. These will not be used.")
if len(actual_categorical_features) != len(categorical_features):
    missing = set(categorical_features) - set(actual_categorical_features)
    print(f"Warning: Missing categorical features from CSV: {missing}. These will not be used.")

# Split data
# Ensure y has enough samples for stratification if some classes are very rare
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as e:
    print(f"Stratify error during train_test_split (likely due to very few samples in a class): {e}")
    print("Attempting split without stratify. Class distribution might be skewed.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


print("\nClass distribution in y_train (before any resampling):")
y_train_counts = Counter(y_train)
print(y_train_counts)
print("\nClass distribution in y_test:")
print(Counter(y_test))

# Create preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, actual_numeric_features), # Use actual features found
        ('cat', categorical_transformer, actual_categorical_features) # Use actual features found
    ], remainder='drop' # Drop any columns not specified, if X wasn't filtered perfectly
)

print("\n--- Training Best Logistic Regression Model for Export ---")

# First, transform the training data using the preprocessor
# The preprocessor should be fit only on X_train
print("Fitting preprocessor on X_train and transforming X_train...")
X_train_processed = preprocessor.fit_transform(X_train)
print(f"Shape of X_train_processed: {X_train_processed.shape}")

# Apply SMOTE to the processed training data
print("\nApplying SMOTE to processed training data...")
min_samples_in_y_train = min(y_train_counts.values()) if y_train_counts else 0

if min_samples_in_y_train < 2 : # SMOTE needs at least 2 samples in the smallest class (k_neighbors default is 5, min is 1)
    print(f"Warning: Smallest class in y_train has {min_samples_in_y_train} sample(s). SMOTE cannot be applied effectively or might error.")
    print("Proceeding to train classifier on original (imbalanced) preprocessed data.")
    X_train_final_for_classifier, y_train_final_for_classifier = X_train_processed, y_train
else:
    # Adjust k_neighbors for SMOTE: it must be less than the number of samples in the smallest class.
    smote_k_neighbors = min(5, min_samples_in_y_train - 1)
    print(f"Using k_neighbors={smote_k_neighbors} for SMOTE.")
    try:
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        X_train_final_for_classifier, y_train_final_for_classifier = smote.fit_resample(X_train_processed, y_train)
        print("Class distribution in y_train_final_for_classifier (after SMOTE):")
        print(Counter(y_train_final_for_classifier))
    except ValueError as e:
        print(f"Error during SMOTE: {e}. Proceeding with original preprocessed data.")
        X_train_final_for_classifier, y_train_final_for_classifier = X_train_processed, y_train


# Define the classifier
lr_classifier = LogisticRegression(class_weight='balanced', # Still useful even with SMOTE
                                   max_iter=1000,
                                   multi_class='multinomial',
                                   solver='lbfgs',
                                   C=0.1, # Regularization strength
                                   random_state=42)

# Train the classifier on the (potentially SMOTE'd) processed data
print("\nTraining Logistic Regression classifier...")
lr_classifier.fit(X_train_final_for_classifier, y_train_final_for_classifier)
print("Logistic Regression classifier trained.")

# Create the final export pipeline: preprocessor (fit on X_train) + classifier (fit on X_train_final_for_classifier)
# The preprocessor is already fit. We just need to ensure the pipeline object has it.
final_export_pipeline = Pipeline([
    ('preprocessor', preprocessor), # This preprocessor was fit on the original X_train
    ('classifier', lr_classifier)   # This classifier was fit on the (SMOTE'd) processed X_train
])

# Quick evaluation on the test set (test data also needs preprocessing by the *fitted* preprocessor)
print("\nEvaluating the final pipeline on the test set...")
X_test_processed = preprocessor.transform(X_test) # Use the preprocessor fit on X_train
accuracy = lr_classifier.score(X_test_processed, y_test) # Evaluate only the classifier part
# Or evaluate the whole pipeline: accuracy = final_export_pipeline.score(X_test, y_test)
print(f"Accuracy of the exported Logistic Regression pipeline on test set: {accuracy:.4f}")


# Save the trained pipeline
model_filename = 'best_logistic_regression_pipeline.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(final_export_pipeline, file)
print(f"\nBest Logistic Regression pipeline saved as '{model_filename}'")

# Save the class order from the trained classifier
model_classes = lr_classifier.classes_.tolist()
classes_filename = 'model_classes.json'
with open(classes_filename, 'w') as f:
    json.dump(model_classes, f) # Save the list of strings/numbers
print(f"Model classes ({model_classes}) saved to '{classes_filename}' for probability mapping.")
print("\nScript finished.")
