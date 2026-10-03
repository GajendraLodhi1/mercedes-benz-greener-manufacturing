# Mercedes-Benz Greener Manufacturing

## 🚗 Project Overview

This project tackles the **Mercedes-Benz Greener Manufacturing** Kaggle challenge. The objective is to predict the exact time (`y` in seconds) that a car will take to pass the testing phase based on its custom configuration.

Since Mercedes-Benz offers a vast array of custom vehicle configurations, ensuring the safety and reliability of each unique car requires robust testing. By accurately predicting testing time, we can optimize the testing process on the test benches, balance workloads, and ultimately reduce CO2 emissions, contributing to greener manufacturing.

## 📊 Dataset

The dataset consists of anonymized features representing different car configurations:
- **`train.csv`**: 4,209 samples and 378 features (including the target `y`).
- **`test.csv`**: 4,209 samples and 377 features (without the target).
- Features include both categorical variables (like `X0` to `X8`) and numerous binary flags.

## ⚙️ Methodology

1. **Exploratory Data Analysis (EDA):**
   - Analyzed the distribution of the target variable (testing time).
   - Visualizations are stored in the `figures/` directory.

2. **Data Preprocessing:**
   - Applied **Label Encoding** to transform categorical features (`X0` to `X8`) into a format suitable for machine learning models.
   - Split the training data into 80% training and 20% validation sets.

3. **Model Training & Evaluation:**
   - We treated this as a Regression problem since the target is continuous.
   - Evaluated multiple models including Linear Regression, Random Forest, and XGBoost.
   - Used metrics such as $R^2$, RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error).

## 🏆 Results

The **XGBoost Regressor** achieved the best performance on the validation set.

| Model | $R^2$ Score | RMSE | MAE |
|-------|------------|------|-----|
| Baseline (Mean) | -0.0000 | 12.4762 | 10.1426 |
| Linear Regression | 0.5465 | 8.4019 | 5.6985 |
| Random Forest | 0.4768 | 9.0244 | 6.1309 |
| **XGBoost** | **0.5908** | **7.9809** | **5.3234** |

You can find a more detailed breakdown in the [`model_report.md`](model_report.md).

## 🚀 How to Run

1. Clone this repository.
2. Ensure you have the dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) in the project root directory. You can download them from the [Kaggle Competition Page](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/data).
3. Install the required Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
4. Run the training script:
   ```bash
   python train_model.py
   ```
   This will:
   - Perform EDA and save plots to the `figures/` folder.
   - Train the models and output performance metrics.
   - Generate `model_report.md`.
   - Create a `submission.csv` with predictions on the test set.

## 📁 Repository Structure

```
├── figures/                  # Generated plots (Target distribution, Feature importance)
├── train_model.py            # Main script for data processing, training, and evaluation
├── model_report.md           # Detailed analysis and model evaluation report
├── README.md                 # Project documentation
├── train.csv                 # Training dataset (Not included in repo if large)
├── test.csv                  # Test dataset (Not included in repo if large)
├── sample_submission.csv     # Sample submission format
└── submission.csv            # Generated predictions for Kaggle
```
