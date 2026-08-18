# Machine Learning Classification & Deployment Assignment

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare multiple Machine Learning classification algorithms on a structured tabular dataset (Red Wine Quality Dataset) and deploy an interactive web interface for evaluation using Streamlit Community Cloud.

## b. Dataset Description
- **Dataset**: UCI Red Wine Quality Dataset
- **Instances**: 1,599 instances
- **Features**: 12 numerical features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality index).
- **Target Variable**: Binary Classification (1: Good Quality [Rating >= 6], 0: Bad Quality [Rating < 6]).

## c. GitHub Repository Link
[PASTE_YOUR_ACTUAL_GITHUB_REPOSITORY_URL_HERE](https://github.com/Abhishekns1997/2025AC05022_ML_assignment_2) 

## d. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7438 | 0.8174 | 0.7514 | 0.7765 | 0.7637 | 0.4832 |
| **Decision Tree** | 0.7594 | 0.7583 | 0.7719 | 0.7765 | 0.7741 | 0.5167 |
| **kNN** | 0.7469 | 0.8066 | 0.7586 | 0.7765 | 0.7674 | 0.4897 |
| **Naive Bayes** | 0.7188 | 0.7937 | 0.7222 | 0.7647 | 0.7429 | 0.4326 |
| **Random Forest (Ensemble)** | 0.8188 | 0.8996 | 0.8324 | 0.8294 | 0.8309 | 0.6358 |

## e. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundary yields moderate performance. Effective baseline model but suffers from slight underfitting on non-linear features. |
| **Decision Tree** | Captures non-linear dependencies well, but shows variance sensitivity and slightly lower generalizability compared to ensemble methods. |
| **kNN** | Distance-based algorithm performs well post-feature standardization; sensitive to optimal neighborhood settings (k=5). |
| **Naive Bayes** | Displays lower accuracy due to the strong feature independence assumption, which is violated by correlated chemical attributes. |
| **Random Forest (Ensemble)** | Outperforms single models by reducing variance through bagging and decision tree aggregation. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** achieved the highest accuracy (0.8188), AUC (0.8996), F1 Score (0.8309), and MCC (0.6358). |
