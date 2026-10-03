import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import os

print("Starting analysis and model training for Mercedes-Benz Greener Manufacturing...")

# 1. Load Data
try:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(f"Loaded train.csv with shape: {train.shape}")
    print(f"Loaded test.csv with shape: {test.shape}")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

# 2. EDA and plots
os.makedirs("figures", exist_ok=True)

# Target Distribution Plot
plt.figure(figsize=(10, 6))
sns.histplot(train["y"], bins=50, kde=True, color="blue")
plt.xlabel("Test Time (y)")
plt.ylabel("Frequency")
plt.title("Distribution of Test Time")
plt.savefig("figures/target_distribution.png")
plt.close()

# 3. Preprocessing
# Identify categorical columns
cat_cols = [col for col in train.columns if train[col].dtype == 'object']

print(f"Encoding categorical features: {cat_cols}")

# Combine train and test for consistent encoding
df_all = pd.concat([train.drop(['y'], axis=1), test], axis=0)

# Label Encoding
for col in cat_cols:
    le = LabelEncoder()
    # Fit on all possible values and handle unseen ones
    df_all[col] = le.fit_transform(df_all[col].astype(str))

# Separate back into train and test
X_train_full = df_all.iloc[:train.shape[0], :]
X_test = df_all.iloc[train.shape[0]:, :]
y_train_full = train['y']

# Drop the ID column for training
X_train_full_features = X_train_full.drop(['ID'], axis=1)
X_test_features = X_test.drop(['ID'], axis=1)

# 4. Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_features, y_train_full, test_size=0.2, random_state=42
)

# 5. Baseline and Model Training
results = []

def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return {"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae, "model_obj": model}

# Baseline (Mean)
mean_pred = np.full_like(y_val, y_train.mean())
baseline_r2 = r2_score(y_val, mean_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_val, mean_pred))
baseline_mae = mean_absolute_error(y_val, mean_pred)
print(f"Baseline (Mean) -> R2: {baseline_r2:.4f}, RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")
results.append({"Model": "Baseline (Mean)", "R2": baseline_r2, "RMSE": baseline_rmse, "MAE": baseline_mae})

# Linear Regression
lr = LinearRegression()
results.append(evaluate_model("Linear Regression", lr, X_train, y_train, X_val, y_val))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
results.append(evaluate_model("Random Forest", rf, X_train, y_train, X_val, y_val))

# XGBoost
xgbr = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1)
results.append(evaluate_model("XGBoost", xgbr, X_train, y_train, X_val, y_val))

# 6. Feature Importance (XGBoost)
best_model_dict = [res for res in results if res['Model'] == 'XGBoost'][0]
best_model = best_model_dict['model_obj']

feature_importances = best_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png")
plt.close()

# 7. Generate Report
report_content = f"""# Mercedes-Benz Greener Manufacturing Analysis Report

## 1. Project Overview
The objective of this project is to predict the testing time (`y`) of different Mercedes-Benz vehicle configurations. Since the target variable is continuous (time in seconds), this is a **Regression problem**, not a classification problem. Therefore, we use regression metrics like $R^2$, RMSE, and MAE instead of a confusion matrix.

## 2. Exploratory Data Analysis
- **Training set shape:** {train.shape}
- **Test set shape:** {test.shape}
- The data contains anonymized categorical features (X0 to X8) and numerous binary features. 
- The target distribution plot was generated and saved to `figures/target_distribution.png`.

## 3. Preprocessing
- Handled categorical features by applying Label Encoding.
- Split the dataset into 80% training and 20% validation.

## 4. Model Evaluation Results
We built multiple models starting from a simple baseline to complex tree-based ensembles.

| Model | $R^2$ Score | RMSE | MAE |
|-------|------------|------|-----|
"""

for res in results:
    report_content += f"| {res['Model']} | {res['R2']:.4f} | {res['RMSE']:.4f} | {res['MAE']:.4f} |\n"

report_content += """
**Metric Interpretations:**
- **$R^2$ (R-squared):** Represents the proportion of variance in the test time explained by the features. Higher is better.
- **RMSE (Root Mean Squared Error):** The average deviation of the predictions from the actual test times in seconds. Lower is better.
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual test times. Lower is better.

## 5. Feature Importance
The top 15 features were analyzed using the best performing model (XGBoost) to understand what variables strongly influence testing time. The visualization is saved as `figures/feature_importance.png`.

## 6. Conclusion
By predicting the test time of a vehicle configuration, we can optimize the testing process, balance workloads on the test benches, and reduce CO2 emissions by identifying configurations that might require excessive testing time.
"""

with open("model_report.md", "w") as f:
    f.write(report_content)
print("Report generated: model_report.md")

# 8. Train on Full Data and Create Submission
print("Training final XGBoost model on full dataset for submission...")
best_model.fit(X_train_full_features, y_train_full)
test_preds = best_model.predict(X_test_features)

submission = pd.DataFrame({
    'ID': X_test['ID'],
    'y': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
