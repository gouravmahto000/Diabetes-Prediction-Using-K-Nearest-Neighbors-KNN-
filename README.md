
# Diabetes Prediction Using K-Nearest Neighbors (KNN)

## Project Overview

This project predicts whether a patient is likely to have diabetes using the **K-Nearest Neighbors (KNN)** machine learning algorithm.

The model is trained on the **Pima Indians Diabetes Dataset**, which contains medical information such as glucose level, blood pressure, BMI, age, and insulin level.

The primary objective is to build a classification model that can accurately predict diabetes for new patients based on their medical measurements.

---

# Business Problem

Hospitals and healthcare organizations collect a large amount of patient data.

Doctors often need a quick prediction of whether a patient is at risk of diabetes.

A machine learning model can assist healthcare professionals by:

* Detecting diabetes early
* Supporting clinical decision making
* Reducing manual analysis
* Improving patient care

---

# Project Objective

The objective of this project is to:

* Clean the medical dataset
* Perform exploratory data analysis
* Prepare the data for machine learning
* Train a KNN classification model
* Evaluate model performance
* Predict diabetes for new patients
* Save the trained model for future use

---

# Dataset Information

Dataset: Pima Indians Diabetes Dataset

Target Variable:

**Outcome**

* 0 → No Diabetes
* 1 → Diabetes

Features:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

---

# Project Workflow

```
Import Libraries
       ↓
Load Dataset
       ↓
Understand Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Train KNN Model
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
ROC Curve
       ↓
Save Model
       ↓
Predict New Patient
```

---

# Why Each Step is Used

## 1. Import Libraries

Purpose:

Import all required Python libraries for data manipulation, visualization, machine learning, and model evaluation.

Libraries Used:

* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* joblib

---

## 2. Load Dataset

Purpose:

Read the dataset into a Pandas DataFrame for analysis.

```python
pd.read_csv()
```

---

## 3. Explore Dataset

Purpose:

Understand the dataset before building the model.

Methods Used:

* head()
* tail()
* sample()
* shape
* info()
* describe()

These functions help identify:

* Number of rows
* Number of columns
* Data types
* Statistical summary

---

## 4. Check Missing Values

Purpose:

Identify missing data.

```python
df.isnull().sum()
```

Although the dataset has no actual NULL values, several columns contain **0**, which is medically impossible.

---

## 5. Replace Invalid Values

Columns:

* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI

Purpose:

Replace invalid zeros with NaN because these measurements cannot realistically be zero.

```python
replace(0, np.nan)
```

---

## 6. Fill Missing Values

Purpose:

Machine learning algorithms cannot work with missing values.

Median is used because it is robust to outliers.

```python
fillna(df.median())
```

---

## 7. Remove Duplicate Rows

Purpose:

Duplicate records can bias the model and reduce its ability to generalize.

```python
drop_duplicates()
```

---

## 8. Exploratory Data Analysis (EDA)

Purpose:

Understand the relationships and distributions within the data.

Visualizations Used:

* Histogram
* Boxplot
* Correlation Heatmap
* Count Plot

These help identify:

* Outliers
* Feature distributions
* Correlations
* Class imbalance

---

## 9. Feature Selection

Purpose:

Separate independent variables from the target variable.

```python
X = Features

y = Target
```

---

## 10. Train-Test Split

Purpose:

Split the data into training and testing sets.

Training Data:

Used to train the model.

Testing Data:

Used to evaluate the model on unseen data.

```python
train_test_split()
```

Why use `stratify=y`?

To maintain the same proportion of diabetic and non-diabetic cases in both training and testing datasets.

---

## 11. Feature Scaling

Purpose:

KNN calculates distances between data points.

If features have different scales, larger-valued features dominate the distance calculation.

StandardScaler standardizes all features to a similar scale.

```python
StandardScaler()
```

Without scaling, KNN performance decreases significantly.

---

## 12. Build KNN Model

Purpose:

Train the classifier using neighboring data points.

```python
KNeighborsClassifier()
```

Default:

```python
n_neighbors = 5
```

This means the model looks at the five nearest neighbors before assigning a class.

---

## 13. Train Model

Purpose:

Allow the model to learn patterns from historical data.

```python
fit()
```

---

## 14. Prediction

Purpose:

Predict diabetes status for unseen patients.

```python
predict()
```

Output:

* 0
* 1

---

## 15. Accuracy Score

Purpose:

Measure overall prediction accuracy.

Formula:

```
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

---

## 16. Confusion Matrix

Purpose:

Understand prediction performance in detail.

Shows:

* True Positive
* True Negative
* False Positive
* False Negative

Useful for medical diagnosis.

---

## 17. Classification Report

Purpose:

Provides multiple evaluation metrics.

Includes:

* Precision
* Recall
* F1-score
* Support

These metrics provide a better understanding than accuracy alone.

---

## 18. Hyperparameter Tuning

Purpose:

Find the best value of **K**.

Instead of assuming K = 5, we test multiple values.

Example:

```
K = 1

K = 2

...

K = 20
```

The model with the highest accuracy is selected.

---

## 19. Accuracy vs K Plot

Purpose:

Visualize how accuracy changes with different K values.

Helps identify the optimal number of neighbors.

---

## 20. ROC Curve

Purpose:

Evaluate classification performance at different thresholds.

The ROC curve plots:

* True Positive Rate
* False Positive Rate

AUC (Area Under Curve):

* 1.0 → Perfect Model
* 0.5 → Random Guess

Higher AUC indicates better performance.

---

## 21. Save Model

Purpose:

Store the trained model for future use.

```python
joblib.dump()
```

Saved Files:

* knn_diabetes.pkl
* scaler.pkl

This avoids retraining the model every time.

---

## 22. Predict New Data

Purpose:

Predict diabetes for a new patient who was not part of the training dataset.

Example:

```
New Patient

↓

Scale Features

↓

Model Prediction

↓

Diabetic or Not
```

This is the actual real-world application of the project.

---

# Machine Learning Concepts Covered

* Data Cleaning
* Missing Value Handling
* Exploratory Data Analysis
* Feature Engineering
* Feature Scaling
* Train-Test Split
* Classification
* KNN Algorithm
* Hyperparameter Tuning
* Model Evaluation
* ROC-AUC Analysis
* Model Persistence
* Real-world Prediction

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

# Expected Outcome

After completing this project, you will be able to:

* Build an end-to-end machine learning classification model.
* Understand how KNN works internally.
* Evaluate classification models using multiple metrics.
* Select the best hyperparameters.
* Save and deploy trained models.
* Predict outcomes for new patient records.

---

# Conclusion

This project demonstrates a complete machine learning workflow for healthcare prediction using the K-Nearest Neighbors algorithm. It includes data preprocessing, exploratory analysis, model building, hyperparameter tuning, evaluation, and deployment-ready model saving. The final model can be used to assist in predicting diabetes risk for new patients based on their medical information.
