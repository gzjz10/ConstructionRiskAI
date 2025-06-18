# Construction Risk Prediction Model Results

## Dataset Information
- Total samples: 2000
- Class distribution: {'Medium': 1600, 'High': 400}
- Features used: 23

## Model Performance Comparison
| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Logistic Regression | 0.9875 | 0.9809 | 0.9876 |
| Decision Tree | 0.7275 | 0.6492 | 0.7486 |
| Balanced Random Forest | 0.8425 | 0.7921 | 0.8535 |
| RUSBoost | 0.9175 | 0.8822 | 0.9209 |
| Logistic Regression with SMOTE | 0.9750 | 0.9627 | 0.9755 |
| Threshold-Tuned Model | 0.8100 | 0.4945 | 0.7341 |

## Best Model Performance
- **Best Model**: Logistic Regression
- **Macro F1 Score**: 0.9809
- **Weighted F1 Score**: 0.9876
- **Accuracy**: 0.9875

### Class-specific Performance
| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| High | 0.9412 | 1.0000 | 0.9697 |
| Medium | 1.0000 | 0.9844 | 0.9921 |

## Key Observations
1. **Imbalanced Nature**: The dataset showed class imbalance, which was addressed using techniques like class weighting, SMOTE, and specialized algorithms.
2. **Model Selection**: Multiple models were trained and evaluated with a focus on macro F1 score to ensure good performance across all risk categories.
3. **Regularization**: Strong regularization was applied to prevent overfitting on this dataset.
4. **Threshold Tuning**: Adjusting decision thresholds improved recall for minority classes.
