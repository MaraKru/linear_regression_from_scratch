# Comparing Linear Models with Regularization: From Basic Features to Overfitting

## About the Project

A research study in Machine Learning where I compared linear models with L1 and L2 regularization (Lasso, Ridge, ElasticNet) for predicting apartment rental prices. Special attention was given to the effect of feature normalization and overfitting when using high-degree polynomial features. 

### The Task

Predict apartment rental price based on listing features.
Dataset: Kaggle - Two Sigma Connect (Rental Listing Inquiries)
(https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data)

### Getting Started

Unzip the provided archives: test.zip and train.zip

### What I did

1) Data preprocessing
* cleaned the `features` 
* extracted the top 20 most frequent amenities
* created 20 binary features
* applied log transformation to price 

2) Models implemented from scratch
* Linear Regression (SGD, deterministic / stochastic)
* Ridge (L2)
* Lasso (L1)
* ElasticNet (L1 + L2)

3) Experiments
* Compared my implementation with sklearn 
* Feature normalization: MinMaxScaler and StandardScaler
* Training on polynomial features (degree 10, 65 features)
* Baseline models (mean / median)


## Results

### Best model

**Lasso + MinMaxScaler**
- MAE (test): 0.225
- RMSE (test): 0.305
- R2 (test): 0.500

Stable on both train and test. Lasso performed automatic feature selection (many weights became zero)

### Normalization

MinMaxScaler outperformed StandardScaler, especially for Lasso. 

### Polynomial features - catastrophic overfitting

At degree 10, models started predicting prices with millions of dollars in error. Ridge (test, MAE) - 51; Lasso (test, MAE) - 21032; R2 became negative


### Tech stack
* Python 3 + Jupyter Notebook
* pandas, numpy
* scikit-learn (Linear Regression, Ridge, Lasso, ElasticNet, PolynomialFeatures, MinMaxScaler, StandartScaler)
* metrics: MAE, RMSE, R2

### What I learned

! Normalization is critical for gradient descent and regularization

! High-degree polynomial features are a trap

! Log transformation of the target variable helps when prices have a wide spread