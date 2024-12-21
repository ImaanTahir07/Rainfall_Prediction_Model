
# Rainfall Prediction Model

## Overview
This project implements a machine learning model to predict rainfall based on meteorological data. Using the Random Forest Classifier, the model is trained to analyze features from the dataset and classify or predict rainfall occurrences.

## Features
- Data preprocessing and feature engineering.
- Exploratory Data Analysis (EDA) with visualizations.
- Machine learning model training and evaluation.
- Model serialization for deployment using `pickle`.

## Dataset
The dataset used in this project is named `Rainfall.csv`, containing meteorological data. Features include various weather parameters used to predict rainfall. The dataset is loaded and processed using `pandas`.

## Tools and Technologies
- **Python Libraries**:
  - Data Manipulation: `numpy`, `pandas`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Model Serialization: `pickle`
- **Algorithms**:
  - Random Forest Classifier

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ImaanTahir07/Rainfall_Prediction_Model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd rainfall-prediction
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the dataset by placing `Rainfall.csv` in the root directory.
2. Run the Jupyter Notebook `Rainfall_Prediction_Model.ipynb` to:
   - Preprocess the data.
   - Train the model.
   - Evaluate its performance.
   - Save the trained model for deployment.

## Key Steps in the Notebook
1. **Import Dependencies**:
   Essential libraries for data manipulation, visualization, and model building.
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.ensemble import RandomForestClassifier
   ```

2. **Data Loading**:
   Load and inspect the dataset using `pandas`.
   ```python
   data = pd.read_csv("Rainfall.csv")
   data.head()
   ```

3. **Data Visualization**:
   Use `matplotlib` and `seaborn` for EDA to identify patterns and correlations.

4. **Model Training**:
   Train a `RandomForestClassifier` on the preprocessed data.
   ```python
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```

5. **Evaluation**:
   Assess model performance using metrics like accuracy and confusion matrix.
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   print(confusion_matrix(y_test, y_pred))
   print(classification_report(y_test, y_pred))
   ```

6. **Model Serialization**:
   Save the trained model using `pickle` for deployment.
   ```python
   import pickle
   with open('model.pkl', 'wb') as file:
       pickle.dump(model, file)
   ```

## Results
The Random Forest Classifier achieved satisfactory performance in predicting rainfall based on the evaluation metrics.

- Libraries: `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`.
