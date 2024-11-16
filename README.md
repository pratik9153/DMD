# Diamond Price Prediction

## Introduction

This project aims to predict the price of a diamond based on its attributes using regression analysis. The dataset contains information about various diamond properties, such as carat, cut, color, clarity, and dimensions. The goal is to build a model that can accurately estimate the price of a diamond based on these features.

## Dataset Overview

The dataset used in this project is available from Kaggle and contains 10 independent variables and 1 target variable (`price`), which we aim to predict. The variables are:

### Independent Variables:
1. **id**: Unique identifier for each diamond.
2. **carat**: Carat (ct.) is the unit of weight measurement used exclusively to weigh gemstones and diamonds.
3. **cut**: Quality of the diamond's cut (categorized).
4. **color**: The color of the diamond (categorized).
5. **clarity**: The clarity of the diamond, which measures its purity and rarity. Graded based on the visibility of imperfections under 10x magnification (categorized).
6. **depth**: The depth of the diamond, measured from the culet (bottom tip) to the table (top surface), in millimeters.
7. **table**: The size of the top facet of the diamond when viewed face up.
8. **x**: The X dimension of the diamond (in millimeters).
9. **y**: The Y dimension of the diamond (in millimeters).
10. **z**: The Z dimension of the diamond (in millimeters).

### Target Variable:
- **price**: The price of the given diamond (dependent variable).

## Dataset Source

The dataset can be accessed from Kaggle using the following link:  
[Diamond Price Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

## Project Workflow

1. **Data Preprocessing**: 
   - Handling missing values in both categorical and numerical columns.
   - Encoding categorical variables (`cut`, `color`, `clarity`) using appropriate techniques.
   - Scaling numerical features to standardize the data.

2. **Model Development**:
   - Training regression models such as Linear Regression, Lasso, Ridge, Decision Tree Regressor, and others.
   - Evaluating model performance using metrics like RMSE, MAE, and R2 score.

3. **Model Evaluation**:
   - Assessing the trained models based on prediction accuracy.
   - Selecting the best-performing model based on evaluation metrics.

4. **Prediction**: 
   - Once the model is trained, it can be used to predict the price of new diamonds based on their features.


