# Model Training Report

## Best Performing Model
- **Model**: Logistic Regression
- **Accuracy**: 0.832
- **Precision**: 0.800
- **Recall**: 0.754
- **F1-Score**: 0.776

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.832 | 0.800 | 0.754 | 0.776 | 0.862 |
| Support Vector Machine | 0.832 | 0.810 | 0.739 | 0.773 | 0.844 |
| Gradient Boosting | 0.816 | 0.810 | 0.681 | 0.740 | 0.847 |
| K-Nearest Neighbors | 0.804 | 0.758 | 0.725 | 0.741 | 0.841 |
| Random Forest | 0.793 | 0.750 | 0.696 | 0.722 | 0.809 |
| Decision Tree | 0.782 | 0.721 | 0.710 | 0.715 | 0.765 |


## Key Findings

### 1. Best Performing Algorithm
The Logistic Regression achieved the highest accuracy of 0.832 on the test set.

### 2. Model Strengths
- **Tree-based models** (Random Forest, Gradient Boosting) typically perform well on tabular data
- **Logistic Regression** provides good interpretability with reasonable performance
- **Ensemble methods** often outperform single models

### 3. Important Features
Based on feature importance analysis, the most predictive features align with historical knowledge:
- Passenger class and fare (proxy for socioeconomic status)
- Gender (reflecting "women and children first" protocol)
- Age and family characteristics

## Recommendations

1. **Production Model**: Use Logistic Regression for predictions
2. **Interpretability**: Consider Logistic Regression if model explainability is important
3. **Monitoring**: Track model performance over time for potential retraining

## Next Steps
1. Integrate the best model into the Streamlit application
2. Create prediction interface for new passenger data
3. Develop model monitoring system
