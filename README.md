
# E-Commerce Purchase Intention Prediction

## 1. Problem Statement

Many visitors browse e-commerce websites without completing a purchase. Predicting whether a shopping session will result in a purchase can help businesses understand visitor behaviour and make better marketing and customer-engagement decisions.

The objective of this project is to develop an end-to-end predictive analytics system that classifies online shopping sessions into purchase and non-purchase categories using machine learning.

The project evaluates two datasets separately and compares five classification algorithms to understand their performance across different datasets.

## 2. Datasets

### Dataset 1: Clickstream Purchase Propensity

This dataset contains clickstream and behavioural information related to user interactions with an online shopping website.

- Target variable: `ordered`
- `1`: Purchase/order
- `0`: No purchase/order
- Features: Behavioural flags representing user interactions and browsing actions.
- Purpose: Predict purchase propensity using observed clickstream behaviour.

### Dataset 2: Online Shoppers Purchasing Intention

This dataset is the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

- Original records: 12,330 sessions
- Original attributes: 18, including the target
- Target variable: `Revenue`
- `True`: Purchase
- `False`: No purchase
- Features: Page visits, browsing duration, bounce rates, exit rates, page values, visitor type, traffic information and other session characteristics.

The two datasets are evaluated independently because their feature structures and target variables differ.

## 3. Machine Learning Models

The project compares the following five classification algorithms:

1. **Logistic Regression:** A baseline classification algorithm used to model relationships between input features and purchase probability.
2. **Random Forest:** An ensemble of decision trees that captures nonlinear relationships and feature interactions.
3. **Gradient Boosting:** A sequential boosting algorithm that builds trees to improve predictions by learning from previous errors.
4. **XGBoost:** An optimized gradient-boosting implementation with additional regularization and computational improvements.
5. **Support Vector Machine (SVM):** A classification algorithm that separates classes by learning a decision boundary.

All five models are intended to be evaluated on both datasets using a consistent experimental methodology.

## 4. Methodology

### 4.1 Data Understanding and Exploratory Data Analysis

The datasets are examined to understand their structure, feature types, class distribution, duplicate records and relationships between features and purchase outcomes.

### 4.2 Data Cleaning

Duplicate records and missing values are checked. Inconsistent data types are handled where necessary before modelling.

### 4.3 Feature Engineering

Additional behavioural or session-level features may be derived from existing variables to represent browsing activity and user interactions more effectively.

### 4.4 Data Preprocessing

Numerical and categorical variables are prepared for machine learning. Scaling, encoding and missing-value handling are applied as appropriate to the selected algorithms.

### 4.5 Train-Validation-Test Split

The data is divided into training, validation and test subsets using stratified sampling where implemented. The validation data supports model and threshold selection, while the test data is reserved for final evaluation.

### 4.6 Class Imbalance Handling

Purchase sessions are less frequent than non-purchase sessions in these datasets. Class-weighting or other imbalance-handling techniques are applied where implemented in the training workflow.

Any resampling technique must be fitted only on training data to avoid data leakage.

### 4.7 Model Training and Comparison

Logistic Regression, Random Forest, Gradient Boosting, XGBoost and SVM are trained and compared separately on each dataset.

Hyperparameters and classification thresholds may be selected using training and validation data, depending on the implemented workflow.

### 4.8 Model Evaluation

The following metrics are used to compare classification performance:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted purchases that are actual purchases.
- **Recall:** Proportion of actual purchases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to rank purchase sessions above non-purchase sessions.
- **Confusion Matrix:** Counts of correct and incorrect classifications.

Additional plots may include ROC curves, precision-recall curves, model comparison charts, threshold analysis and feature importance.

### 4.9 Model Interpretation

Model results and feature importance are examined to identify variables associated with purchase behaviour. These associations should not automatically be interpreted as causal relationships.

## 5. Development Workflow

1. Define the purchase intention prediction problem.
2. Load both datasets separately.
3. Inspect dataset structure and target distributions.
4. Perform exploratory data analysis.
5. Clean the data and engineer useful features.
6. Apply preprocessing and handle class imbalance.
7. Create stratified data splits.
8. Train five machine learning models on each dataset.
9. Compare validation results and select suitable thresholds.
10. Evaluate the selected configurations on held-out test data.
11. Generate comparison plots and confusion matrices.
12. Interpret the results and identify the best-performing model based on measured performance.

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

The required package versions should be documented in `requirements.txt`.

## 7. How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Verify the datasets

Ensure that both CSV files are present in the expected data directory and that the file paths in the source code match their actual locations.

### Step 5: Run the project

Run the main Python script or open the project's Jupyter Notebook and execute its cells in the required order.

Follow the execution instructions associated with the actual entry-point file in the repository.

## 8. Results

The project produces evaluation metrics and comparison visualizations for five machine learning models across two datasets.

The best-performing model should be selected using the actual validation and held-out test results. Performance figures are not specified here because they must be obtained from the final execution of the project.

## 9. Limitations and Future Scope

- The datasets may not represent every e-commerce website or customer population.
- Performance can vary between datasets because their features and purchase rates differ.
- Threshold selection and class imbalance handling influence precision and recall.
- Future work may include additional hyperparameter optimization, probability calibration, explainability methods and testing on more recent datasets.

## 10. Conclusion

This project demonstrates an end-to-end machine learning workflow for e-commerce purchase intention prediction. By evaluating two datasets independently and comparing five classification algorithms, it aims to identify a suitable predictive model and understand the behavioural factors associated with purchase outcomes.
