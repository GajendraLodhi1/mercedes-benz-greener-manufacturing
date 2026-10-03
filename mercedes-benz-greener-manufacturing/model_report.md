# Mercedes-Benz Greener Manufacturing Analysis Report

## 1. Project Overview
The objective of this project is to predict the testing time (`y`) of different Mercedes-Benz vehicle configurations. Since the target variable is continuous (time in seconds), this is a **Regression problem**, not a classification problem. Therefore, we use regression metrics like $R^2$, RMSE, and MAE instead of a confusion matrix.

## 2. Exploratory Data Analysis
- **Training set shape:** (4209, 378)
- **Test set shape:** (4209, 377)
- The data contains anonymized categorical features (X0 to X8) and numerous binary features. 
- The target distribution plot was generated and saved to `figures/target_distribution.png`.

## 3. Preprocessing
- Handled categorical features by applying Label Encoding.
- Split the dataset into 80% training and 20% validation.

## 4. Model Evaluation Results
We built multiple models starting from a simple baseline to complex tree-based ensembles.

| Model | $R^2$ Score | RMSE | MAE |
|-------|------------|------|-----|
| Baseline (Mean) | -0.0000 | 12.4762 | 10.1426 |
| Linear Regression | 0.5465 | 8.4019 | 5.6985 |
| Random Forest | 0.4768 | 9.0244 | 6.1309 |
| XGBoost | 0.5908 | 7.9809 | 5.3234 |

**Metric Interpretations:**
- **$R^2$ (R-squared):** Represents the proportion of variance in the test time explained by the features. Higher is better.
- **RMSE (Root Mean Squared Error):** The average deviation of the predictions from the actual test times in seconds. Lower is better.
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual test times. Lower is better.

## 5. Feature Importance
The top 15 features were analyzed using the best performing model (XGBoost) to understand what variables strongly influence testing time. The visualization is saved as `figures/feature_importance.png`.

## 6. Conclusion
By predicting the test time of a vehicle configuration, we can optimize the testing process, balance workloads on the test benches, and reduce CO2 emissions by identifying configurations that might require excessive testing time.
