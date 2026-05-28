# House Price Prediction Using Machine Learning

A complete end-to-end regression machine learning project that predicts residential house sale prices using the Ames Housing Dataset from Kaggle.

---

#  Project Overview

This project builds and evaluates multiple regression models to predict house prices based on housing characteristics such as:

* Living area
* Neighborhood
* Garage size
* Year built
* Overall house quality
* Lot size
* Number of rooms
---

#  Objectives

The main goals of this project are to:

* Understand the factors affecting house prices
* Build accurate regression models
* Compare multiple machine learning algorithms
* Tune hyperparameters for improved performance
* Evaluate models using RMSE, MAE, and R² metrics

---

# Dataset

Dataset Source:

* Kaggle: House Prices — Advanced Regression Techniques

Dataset Files:

| File        | Description                         |
| ----------- | ----------------------------------- |
| `train.csv` | Training dataset with target values |
| `test.csv`  | Test dataset without target values  |

Target Variable:

```python
SalePrice
```

The target represents the final sale price of the house in USD.

---

# Technologies Used

## Programming Language

* Python

## Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost

---

#  Exploratory Data Analysis (EDA)

The notebook performs extensive EDA including:

* Sale price distributions
* Log-transformed target analysis
* Correlation heatmaps
* Scatter plots
* Box plots
* Neighborhood analysis
* Residual distributions
* Feature importance analysis

Example insights:

* `OverallQual` strongly correlates with `SalePrice`
* Larger living areas generally increase house prices
* Newer homes tend to have higher sale prices

---

#  Data Preprocessing

The preprocessing pipeline includes:

## Missing Value Handling

* Numeric columns → median imputation
* Categorical columns → mode imputation

## Feature Engineering

Selected important features such as:

```python
[
    'MSSubClass',
    'MSZoning',
    'LotArea',
    'Neighborhood',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'GrLivArea',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd',
    'GarageCars',
    'GarageArea'
]
```

## Encoding

* One-hot encoding using `pd.get_dummies()`

---

#  Machine Learning Models

Three regression models were trained and compared.

| Model                   | Type              |
| ----------------------- | ----------------- |
| Linear Regression       | Baseline          |
| Random Forest Regressor | Ensemble          |
| XGBoost Regressor       | Gradient Boosting |

---

#  Hyperparameter Tuning

Hyperparameter optimization was performed using:

```python
GridSearchCV
```

## Random Forest Parameters

* `n_estimators`
* `max_depth`
* `min_samples_split`

## XGBoost Parameters

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`

---

#  Model Evaluation Metrics

The following metrics were used:

| Metric   | Description                  |
| -------- | ---------------------------- |
| RMSE     | Root Mean Squared Error      |
| MAE      | Mean Absolute Error          |
| R² Score | Coefficient of Determination |

---

#  Best Model

##  XGBoost Regressor

XGBoost achieved:

* Lowest RMSE
* Highest R² score
* Best generalization performance

The model also showed strong performance during cross-validation.

---

#  Visualizations Included

The project contains rich visualizations including:

* Target distribution plots
* Correlation heatmaps
* Actual vs predicted plots
* Residual plots
* Residual distributions
* Feature importance charts
* Cross-validation RMSE plots

---

#  Key Findings

* `OverallQual` is the strongest predictor of sale price
* `GrLivArea` significantly impacts pricing
* XGBoost outperformed all other models
* Log transformation can improve residual normality

---

#  Potential Improvements

Future enhancements may include:

* Target log transformation
* Advanced feature engineering
* Stacking and blending models
* KNN/MICE imputation
* Outlier removal
* Feature scaling optimization
* Deep learning regression models

---

# Project Structure

```bash
House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── images/
│   └── visualizations/
│
├── README.md
└── requirements.txt
```
