# Laptop Price Prediction & Market Analysis 

A professional Machine Learning implementation designed to predict laptop market prices using high-dimensional hardware specifications.

## Executive Summary
This project implements an end-to-end Machine Learning pipeline to analyze and predict laptop pricing. By leveraging advanced data cleaning and ensemble modeling, the system identifies the key hardware drivers that influence market valuation.

## 🛠️ Technical Stack
- **Core:** Python 3.x
- **Data Engines:** Pandas (Data Manipulation), NumPy (Numerical Computing)
- **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest)
- **Visualization:** Matplotlib, Seaborn
- **Preprocessing:** Standard Scaling, One-Hot Encoding, Regex-based Feature Extraction

## ⚙️ Engineering Workflow
- **Advanced Cleaning:** Automated removal of non-numeric units and data type optimization.
- **Feature Engineering:** - Extracted **Display Resolution** (X and Y pixels) and **Touchscreen** capability using Regex.
  - Simplified complex **CPU/GPU** nomenclature into high-impact categorical features.
- **Pipeline:** Implemented a robust preprocessing pipeline to handle multi-collinearity and feature scaling.

## 📈 Model Performance & Benchmarking
The project benchmarked multiple algorithms to optimize predictive accuracy:

| Model | MAE | RMSE |
| :--- | :--- | :--- |
| Linear Regression | 12,434 | 18,152 |
| **Random Forest Regressor** | **10,671** | **17,658** |

**Result:** The Random Forest Regressor reduced the Mean Absolute Error (MAE) by **1,763 units**, demonstrating superior handling of non-linear pricing trends and high-end hardware configurations.

## 📂 Dataset
The data used for this project can be found here: [Dataset Link](https://raw.githubusercontent.com/Raghavagr/Laptop_Price_Prediction/main/laptop_data.csv)
