# 🏥 Medical Cost Personal Dataset – Regression Project

This project applies different regression models (Linear, Ridge, and Lasso) to predict individual medical insurance costs using the **Kaggle Medical Cost Personal Dataset**.  
The workflow follows a structured machine learning approach, from data exploration to model evaluation and visualization.

---

## 📌 Objectives

- Understand the relationship between demographic and lifestyle factors (age, BMI, smoking, etc.) and medical charges.  
- Build a preprocessing and modeling pipeline using **scikit-learn**.  
- Compare regularized regression models (**Ridge**, **Lasso**) with the baseline **Linear Regression**.  
- Evaluate model performance using regression metrics and visualizations.

---

## ⚙️ Workflow

### Block 1: Data Loading & Exploration
- Loaded the dataset from Kaggle (`insurance.csv`).  
- Inspected dataset structure, types, and descriptive statistics.  
- Visualized key variables using histograms and scatter plots:
  - Age distribution  
  - BMI distribution  
  - Charges by smoker/non-smoker  
  - Pairwise relationships with `sns.pairplot()`  

### Block 2: Preprocessing
- Encoded categorical variables (`sex`, `smoker`, `region`).  
- Built a preprocessing pipeline with **ColumnTransformer**:
  - `StandardScaler` for numerical features  
  - `OneHotEncoder` for categorical features  
- Split data into **train/test sets (80/20)**.

### Block 3: Model Training
- Implemented and trained three regression models:
  - **Linear Regression** (baseline)
  - **Ridge Regression** (L2 regularization)
  - **Lasso Regression** (L1 regularization)
- Used pipelines to combine preprocessing and model steps.  
- Tuned hyperparameters (`alpha`) for Ridge and Lasso using **cross-validation**.

### Block 4: Evaluation & Visualization
- Evaluated each model using:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² (Coefficient of Determination)**
- Compared predicted vs. actual charges using scatter plots.
- Visualized model coefficients to analyze feature importance.

---

## 📊 Results

| Model              | Train R² | Test R² | Notes |
|--------------------|----------|---------|-------|
| Linear Regression  | ~0.75    | ~0.73   | Baseline model |
| Ridge Regression   | ~0.76    | ~0.74   | Improved stability |
| Lasso Regression   | ~0.76    | ~0.73   | Similar performance, performs feature selection |

**Key Insights:**
- **Smoker status**, **age**, and **BMI** are the strongest predictors of medical costs.  
- **Regularization (Ridge/Lasso)** helps reduce overfitting and improves model generalization.  
- Lasso performs automatic feature selection by reducing less relevant coefficients to zero.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your_username>/project_04_medical_cost_personal_datasets.git
