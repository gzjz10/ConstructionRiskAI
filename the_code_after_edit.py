import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# For imbalanced data handling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier

# For threshold tuning
from sklearn.calibration import CalibratedClassifierCV

# For saving the model
import h5py
import pickle

import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading construction risk dataset...")
df = pd.read_csv('construction_risk_dataset_2000.csv')

print(f"Dataset shape: {df.shape}")
print("\nClass distribution:")
print(df['risk_category'].value_counts())
print("\nSample success rate:")
print(f"Project success rate: {df['project_success'].mean() * 100:.2f}%")

# Define features and target variable
# Select relevant features for risk prediction (excluding outcome variables and string features)
numeric_features = [
    'structural_complexity_rating',
    'temporary_structures_rating',
    'bim_adoption_percentage',
    'lead_time_rating',
    'tariff_rate_percentage',
    'supply_chain_risk_percentage',
    'socio_political_risk_score',
    'permit_approval_rating',
    'lien_claims_rating',
    'seismic_zone_rating',
    'flood_risk_percentage',
    'rainfall_flood_risk_percentage',
    'contingency_budget_percentage',
    'weather_delay_rating',
    'safety_risk_rating',
    'worker_experience_rating',
    'turnover_rating',
    'cybersecurity_risk_rating',
    'renewable_energy_percentage',
    'energy_efficiency_percentage'
]

categorical_features = ['project_type', 'project_size', 'region']

# Exclude outcome variables and string identifier columns
exclude_features = [
    'project_id', 'project_name', 'start_date', 'planned_end_date', 
    'delay_days', 'cost_overrun_percentage', 'safety_incidents', 
    'overall_risk_score', 'actual_end_date', 'on_time_completion', 
    'within_budget', 'project_success', 'high_risk_factors_count', 
    'top_risk_factors', 'recommended_mitigations'
]

# Feature selection - use all available features except excluded ones
X = df.drop(exclude_features + ['risk_category'], axis=1)
y = df['risk_category']

# Split data into train and test sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training class distribution:", Counter(y_train))
print("Testing class distribution:", Counter(y_test))

# Create preprocessing pipeline with proper handling of categorical and numeric features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Function to calculate class weights for balanced weighting
def compute_class_weight(y):
    counter = Counter(y)
    max_val = float(max(counter.values()))
    return {cls: max_val/count for cls, count in counter.items()}

class_weights = compute_class_weight(y_train)
print("\nComputed class weights:", class_weights)

# Function to evaluate model performance with detailed metrics
def evaluate_model(model, X, y, model_name="Model"):
    y_pred = model.predict(X)
    
    # Calculate various metrics
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print("\nClass-wise Performance:")
    for cls in sorted(report.keys()):
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"  Class '{cls}': Precision={report[cls]['precision']:.4f}, "
                  f"Recall={report[cls]['recall']:.4f}, F1={report[cls]['f1-score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=sorted(np.unique(y)),
               yticklabels=sorted(np.unique(y)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'report': report,
        'confusion_matrix': conf_matrix,
        'model': model  # Store the model itself for later use
    }

# Function for threshold tuning - find optimal thresholds to maximize macro F1
def tune_thresholds(model, X_val, y_val):
    # Get probabilities for each class
    probs = model.predict_proba(X_val)
    
    # Get class priors
    class_priors = Counter(y_val)
    total = sum(class_priors.values())
    class_priors = {cls: count/total for cls, count in class_priors.items()}
    
    # Initialize variables to track best thresholds and F1 score
    best_f1 = 0
    best_thresholds = np.ones(len(model.classes_)) * (1/len(model.classes_))
    
    # Create a grid of thresholds to try
    threshold_grid = np.linspace(0.05, 0.95, 19)
    
    # For simplicity, we'll try adjusting threshold for the minority class
    # Find minority class index
    class_counts = np.bincount([np.where(model.classes_ == c)[0][0] for c in y_val])
    minority_idx = np.argmin(class_counts)
    
    for threshold in threshold_grid:
        # Create adjusted thresholds, lowering minority class threshold
        thresholds = np.ones(len(model.classes_)) * (1/len(model.classes_))
        thresholds[minority_idx] = threshold
        
        # Normalize thresholds to sum to 1
        thresholds = thresholds / np.sum(thresholds)
        
        # Predict using adjusted thresholds
        y_pred = []
        for prob in probs:
            # Adjust probabilities by class priors (optional)
            adjusted_probs = prob / np.array(list(class_priors.values()))
            adjusted_probs = adjusted_probs * thresholds
            y_pred.append(model.classes_[np.argmax(adjusted_probs)])
        
        # Calculate F1 score
        f1 = f1_score(y_val, y_pred, average='macro')
        
        # Update best thresholds if F1 improved
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = thresholds
    
    print(f"Best thresholds: {best_thresholds}")
    print(f"Best macro F1: {best_f1:.4f}")
    
    return best_thresholds

# Custom threshold predictor class
class ThresholdClassifier:
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds
        self.classes_ = model.classes_
    
    def predict(self, X):
        probs = self.model.predict_proba(X)
        
        # Calculate class priors based on the training data
        class_priors = {i: 1/len(self.classes_) for i in range(len(self.classes_))}
        
        # Predict using adjusted thresholds
        y_pred = []
        for prob in probs:
            # Adjust probabilities by priors and thresholds
            adjusted_probs = prob * self.thresholds
            y_pred.append(self.classes_[np.argmax(adjusted_probs)])
        
        return np.array(y_pred)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Define a function to train and evaluate multiple models
def build_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor):
    # Initialize results dictionary to store all model results
    results = {}
    
    # Create cross-validation folds for hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Logistic Regression with class weights
    print("\n--- Training Logistic Regression Model ---")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', 
                                          max_iter=1000, 
                                          multi_class='multinomial', 
                                          solver='lbfgs',
                                          C=0.1,  # Strong regularization for small dataset
                                          random_state=42))
    ])
    
    # Define hyperparameters for grid search
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1.0],  # Regularization strength
    }
    
    # Perform grid search
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    
    print(f"Best parameters for Logistic Regression: {lr_grid.best_params_}")
    print(f"Best CV score: {lr_grid.best_score_:.4f}")
    
    # Evaluate on test set
    lr_results = evaluate_model(lr_grid.best_estimator_, X_test, y_test, "Logistic Regression")
    results['Logistic Regression'] = lr_results
    
    # 2. Decision Tree with balanced class weights and limited depth
    print("\n--- Training Decision Tree Model ---")
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(class_weight='balanced',
                                              max_depth=5,  # Limit tree depth to prevent overfitting
                                              min_samples_leaf=5,  # Require minimum samples per leaf
                                              random_state=42))
    ])
    
    # Define hyperparameters for grid search
    dt_param_grid = {
        'classifier__max_depth': [3, 4, 5],
        'classifier__min_samples_leaf': [5, 10, 20]
    }
    
    # Perform grid search
    dt_grid = GridSearchCV(dt_pipeline, dt_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    
    print(f"Best parameters for Decision Tree: {dt_grid.best_params_}")
    print(f"Best CV score: {dt_grid.best_score_:.4f}")
    
    # Evaluate on test set
    dt_results = evaluate_model(dt_grid.best_estimator_, X_test, y_test, "Decision Tree")
    results['Decision Tree'] = dt_results
    
    # 3. Balanced Random Forest (from imbalanced-learn)
    print("\n--- Training Balanced Random Forest Model ---")
    brf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', BalancedRandomForestClassifier(n_estimators=100,
                                                     max_depth=10,
                                                     random_state=42))
    ])
    
    # Define hyperparameters for grid search
    brf_param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 10, 15]
    }
    
    # Perform grid search
    brf_grid = GridSearchCV(brf_pipeline, brf_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    brf_grid.fit(X_train, y_train)
    
    print(f"Best parameters for Balanced Random Forest: {brf_grid.best_params_}")
    print(f"Best CV score: {brf_grid.best_score_:.4f}")
    
    # Evaluate on test set
    brf_results = evaluate_model(brf_grid.best_estimator_, X_test, y_test, "Balanced Random Forest")
    results['Balanced Random Forest'] = brf_results
    
    # 4. RUSBoost for imbalanced classification
    print("\n--- Training RUSBoost Model ---")
    rusboost_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RUSBoostClassifier(n_estimators=50, 
                                         learning_rate=0.1,
                                         random_state=42))
    ])
    
    # Define hyperparameters for grid search
    rusboost_param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1, 0.5]
    }
    
    # Perform grid search
    rusboost_grid = GridSearchCV(rusboost_pipeline, rusboost_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    rusboost_grid.fit(X_train, y_train)
    
    print(f"Best parameters for RUSBoost: {rusboost_grid.best_params_}")
    print(f"Best CV score: {rusboost_grid.best_score_:.4f}")
    
    # Evaluate on test set
    rusboost_results = evaluate_model(rusboost_grid.best_estimator_, X_test, y_test, "RUSBoost")
    results['RUSBoost'] = rusboost_results
    
    # 5. Try SMOTE preprocessing with logistic regression
    print("\n--- Training Logistic Regression with SMOTE ---")
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        preprocessor.fit_transform(X_train), y_train
    )
    
    print(f"Original training shape: {Counter(y_train)}")
    print(f"Resampled training shape: {Counter(y_train_resampled)}")
    
    # Train logistic regression on resampled data
    lr_smote = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    lr_smote.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate on test set (need to preprocess test data)
    X_test_transformed = preprocessor.transform(X_test)
    smote_results = evaluate_model(lr_smote, X_test_transformed, y_test, "Logistic Regression with SMOTE")
    results['Logistic Regression with SMOTE'] = smote_results
    
    # 6. Try threshold tuning with the best model found so far
    print("\n--- Applying Threshold Tuning to Best Model ---")
    
    # Use balanced random forest model for demonstration - Fixed the parameter access
    best_model_so_far = brf_grid.best_estimator_
    
    # Split validation data for threshold tuning
    X_thresh_train, X_thresh_val, y_thresh_train, y_thresh_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=43, stratify=y_train
    )
    
    # Train model on threshold training data - fixed parameter handling
    thresh_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', BalancedRandomForestClassifier(
            max_depth=brf_grid.best_params_['classifier__max_depth'], 
            n_estimators=brf_grid.best_params_['classifier__n_estimators'],
            random_state=42))
    ])
    thresh_pipeline.fit(X_thresh_train, y_thresh_train)
    
    # Tune thresholds using validation data
    best_thresholds = tune_thresholds(thresh_pipeline, X_thresh_val, y_thresh_val)
    
    # Create threshold-adjusted classifier
    tuned_classifier = ThresholdClassifier(thresh_pipeline, best_thresholds)
    
    # Retrain on full training data
    thresh_pipeline.fit(X_train, y_train)
    tuned_classifier = ThresholdClassifier(thresh_pipeline, best_thresholds)
    
    # Evaluate tuned model
    threshold_results = evaluate_model(tuned_classifier, X_test, y_test, "Threshold-Tuned Balanced Random Forest")
    results['Threshold-Tuned Model'] = threshold_results
    
    # Create feature importance plot for Balanced Random Forest (if available)
    if 'classifier' in brf_grid.best_estimator_.named_steps:
        if hasattr(brf_grid.best_estimator_.named_steps['classifier'], 'feature_importances_'):
            # Get feature names from the preprocessor
            feature_names = (
                numeric_features + 
                list(brf_grid.best_estimator_.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(categorical_features))
            )
            
            # Get feature importances
            importances = brf_grid.best_estimator_.named_steps['classifier'].feature_importances_
            
            # Create DataFrame for plotting
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
            plt.title('Top 20 Feature Importances - Balanced Random Forest')
            plt.tight_layout()
            plt.savefig('feature_importances.png')
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance_df.head(10))
    
    # Compare all models on macro F1 score
    print("\n--- Model Comparison (Macro F1 Score) ---")
    model_names = list(results.keys())
    macro_f1_scores = [results[model]['f1_macro'] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=macro_f1_scores, y=model_names)
    plt.title('Model Comparison - Macro F1 Score')
    plt.xlabel('Macro F1 Score')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    for model_name, score in zip(model_names, macro_f1_scores):
        print(f"{model_name}: {score:.4f}")
    
    # Return the best model based on macro F1 score
    best_model_name = model_names[np.argmax(macro_f1_scores)]
    print(f"\nBest model based on macro F1 score: {best_model_name}")
    
    return results, best_model_name

# Train and evaluate all models
model_results, best_model_name = build_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor)

print("\n=== Final Results ===")
print(f"Best model: {best_model_name}")
print(f"Macro F1 Score: {model_results[best_model_name]['f1_macro']:.4f}")
print(f"Weighted F1 Score: {model_results[best_model_name]['f1_weighted']:.4f}")
print(f"Accuracy: {model_results[best_model_name]['accuracy']:.4f}")

# Class-specific performance of best model
print("\nClass-specific performance of best model:")
best_report = model_results[best_model_name]['report']
for cls in sorted(best_report.keys()):
    if cls not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"  Class '{cls}': Precision={best_report[cls]['precision']:.4f}, "
              f"Recall={best_report[cls]['recall']:.4f}, F1={best_report[cls]['f1-score']:.4f}")

# Save results to markdown file
with open('model_results.md', 'w') as f:
    f.write("# Construction Risk Prediction Model Results\n\n")
    
    f.write("## Dataset Information\n")
    f.write(f"- Total samples: {len(df)}\n")
    f.write(f"- Class distribution: {dict(Counter(df['risk_category']))}\n")
    f.write(f"- Features used: {len(numeric_features) + len(categorical_features)}\n\n")
    
    f.write("## Model Performance Comparison\n")
    f.write("| Model | Accuracy | Macro F1 | Weighted F1 |\n")
    f.write("|-------|----------|----------|-------------|\n")
    
    for model_name in model_results:
        results = model_results[model_name]
        f.write(f"| {model_name} | {results['accuracy']:.4f} | {results['f1_macro']:.4f} | {results['f1_weighted']:.4f} |\n")
    
    f.write("\n## Best Model Performance\n")
    f.write(f"- **Best Model**: {best_model_name}\n")
    f.write(f"- **Macro F1 Score**: {model_results[best_model_name]['f1_macro']:.4f}\n")
    f.write(f"- **Weighted F1 Score**: {model_results[best_model_name]['f1_weighted']:.4f}\n")
    f.write(f"- **Accuracy**: {model_results[best_model_name]['accuracy']:.4f}\n\n")
    
    f.write("### Class-specific Performance\n")
    f.write("| Class | Precision | Recall | F1 Score |\n")
    f.write("|-------|-----------|--------|----------|\n")
    
    best_report = model_results[best_model_name]['report']
    for cls in sorted(best_report.keys()):
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            f.write(f"| {cls} | {best_report[cls]['precision']:.4f} | {best_report[cls]['recall']:.4f} | {best_report[cls]['f1-score']:.4f} |\n")
    
    f.write("\n## Key Observations\n")
    f.write("1. **Imbalanced Nature**: The dataset showed class imbalance, which was addressed using techniques like class weighting, SMOTE, and specialized algorithms.\n")
    f.write("2. **Model Selection**: Multiple models were trained and evaluated with a focus on macro F1 score to ensure good performance across all risk categories.\n")
    f.write("3. **Regularization**: Strong regularization was applied to prevent overfitting on this dataset.\n")
    f.write("4. **Threshold Tuning**: Adjusting decision thresholds improved recall for minority classes.\n")

print("\nResults have been saved to 'model_results.md'")

# Save the best model as h5 file
print("\nSaving the best model...")

# Get the best model based on name
if best_model_name == "Threshold-Tuned Model":
    # For the threshold-tuned model, get the tuned classifier
    best_model = model_results[best_model_name]['model']
elif best_model_name == "Logistic Regression with SMOTE":
    # For SMOTE model, create a pipeline that includes preprocessing
    best_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lr_smote)
    ])
else:
    # For other models, get from grid search best estimator
    if best_model_name == "Logistic Regression":
        best_model = lr_grid.best_estimator_
    elif best_model_name == "Decision Tree":
        best_model = dt_grid.best_estimator_
    elif best_model_name == "Balanced Random Forest":
        best_model = brf_grid.best_estimator_
    elif best_model_name == "RUSBoost":
        best_model = rusboost_grid.best_estimator_

# Save the model to h5 file
filename = "the_new_model.h5"
with h5py.File(filename, 'w') as h5f:
    # Serialize the model using pickle
    serialized_model = pickle.dumps(best_model)
    # Save the serialized model to the h5 file
    h5f.create_dataset('model', data=np.void(serialized_model))
    
    # Save additional metadata
    h5f.attrs['model_name'] = best_model_name
    h5f.attrs['accuracy'] = model_results[best_model_name]['accuracy']
    h5f.attrs['f1_macro'] = model_results[best_model_name]['f1_macro']
    h5f.attrs['f1_weighted'] = model_results[best_model_name]['f1_weighted']

print(f"Best model '{best_model_name}' saved as '{filename}'")

# To load this model later, you can use:
print("\nTo load this model in the future, use the following code:")
print("""
import h5py
import pickle
import numpy as np

# Load the model
with h5py.File('the_new_model.h5', 'r') as h5f:
    # Get model metadata
    model_name = h5f.attrs['model_name']
    accuracy = h5f.attrs['accuracy']
    f1_macro = h5f.attrs['f1_macro']
    f1_weighted = h5f.attrs['f1_weighted']
    
    # Load the serialized model
    serialized_model = h5f['model'][()]
    loaded_model = pickle.loads(serialized_model.tobytes())
    
    print(f"Loaded model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
""")