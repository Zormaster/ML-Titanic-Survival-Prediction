# ML-Titanic-Survival-Prediction
The goal of this project is to gain introductory experience with Machine Learning and refresh python skills by building a predictive model for Titanic survival unassisted by AI coding assistants. Acceptable use of AI assistance includes generative search tools, single-line auto-complete, generated initial project instructions, and summarized results.

## Initial Instructions

Here is a suggestion for a classic, beginner-friendly project that touches upon the entire ML workflow using popular Python libraries

### 💡 Project Suggestion: Titanic Survival Prediction
This project uses the famous **Titanic dataset**, a classic supervised classification problem.

| Feature | Detail |
| --- | --- |
| **Problem Type** | Binary Classification (Survived: 1 or Died: 0) |
| **Goal** | Predict whether a passenger survived the Titanic sinking based on their features. |
| **Core Libraries** | **Pandas** (Data Handling), **Scikit-learn** (ML Algorithm), **Matplotlib/Seaborn** (Visualization). |
| **Suggested Algorithm** | **Logistic Regression** (Simple, effective baseline classifier). |

---

### 5-Step Implementation Plan
#### Step 1: Data Acquisition & Initial Exploration
1. **Get the Data:** Download the `train.csv` dataset (often available directly through Kaggle or many Scikit-learn tutorials).
2. **Load Data:** Use Pandas to load the CSV file into a DataFrame.
```python
import pandas as pd
df = pd.read_csv('train.csv')

```


3. **Explore:** Check the data for types, missing values, and the distribution of the target variable (`Survived`).
* *Observation:* You will quickly see missing values in features like `Age` and `Cabin`.



#### Step 2: Data Preprocessing (Cleaning & Feature Engineering)
1. **Handle Missing Values:**
* Fill missing `Age` values with the median age.
* Fill missing `Embarked` (port of embarkation) values with the mode (most common port).
* Drop the `Cabin` column due to too many missing values.


2. **Drop Irrelevant Features:** Drop columns like `PassengerId`, `Name`, and `Ticket`, as they are usually not good predictors for the model.
3. **Encode Categorical Features:** Machine learning algorithms work with numbers, so categorical text features must be converted.
* Convert `Sex` (Male/Female) to numerical (0/1).
* Convert `Embarked` (C/Q/S) using **one-hot encoding** (e.g., using `pd.get_dummies()`).



#### Step 3: Model Selection & Training
1. **Define Features (X) and Label (y):**
* X will be your preprocessed features (`Pclass`, `Sex_encoded`, `Age`, `Fare`, `Embarked_C`, etc.).
* y will be the target column (`Survived`).


2. **Split Data:** Split X and y into training and testing sets (e.g., 80% train, 20% test).
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```


3. **Train the Model:** Instantiate and train a Logistic Regression model.
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

```



#### Step 4: Evaluation
1. **Make Predictions:** Use the trained model to predict survival on the held-out test set.
```python
y_pred = model.predict(X_test)

```


2. **Evaluate Performance:** Calculate the key classification metrics.
```python
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```


* *Tip:* Check the **Precision** and **Recall** values to see if the model is good at catching survivors (Recall) or if its positive predictions are reliable (Precision).



#### Step 5: Iteration and Improvement (Optional, but encouraged)
**Try Different Algorithms:** See if a **Decision Tree** or **Random Forest** can beat your Logistic Regression accuracy.
* **Hyperparameter Tuning:** Use methods like `GridSearchCV` to find the optimal settings for your chosen algorithm.
* **Feature Engineering:** Try creating a new feature, like the size of the family (combining `SibSp` and `Parch`), and see if it improves your score.

This project will give you hands-on experience with every stage of the ML lifecycle!

---

## Titanic Survival Prediction - Results Summary

## Dataset Preprocessing
- **Missing Values Handled:**
  - Age: Filled with median (28.0 years)
  - Embarked: Filled with mode ('S' - Southampton)
- **Feature Engineering:** One-hot encoding applied to Sex and Embarked columns
- **Dropped Columns:** PassengerId, Name, Ticket, Cabin (non-predictive or too many missing values)


## Model Performance Comparison

### Baseline Models (Default Hyperparameters)
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 79.33% |
| Random Forest | **80.45%** ⭐ |
| Support Vector Machine | 78.21% |
| Decision Tree | 77.65% |

**Initial Winner:** Random Forest (80.45%)

---

### After Hyperparameter Tuning

| Model | Accuracy | Best Hyperparameters | Improvement |
|-------|----------|---------------------|-------------|
| **Random Forest** | **81.01%** ⭐ | gini, max_depth=5, min_samples_split=10, n_estimators=10 | +0.56% |
| Decision Tree | 79.89% | entropy, max_depth=3, min_samples_split=2 | +2.24% |
| Logistic Regression | 79.33% | C=1, penalty='l1', solver='liblinear' | 0% |

**Best Model After Tuning:** Random Forest (81.01%)



## Feature Engineering Results

### Attempt 1: Added Family-Based Features
- **New Features:** `FamilySize` (SibSp + Parch + 1) and `IsAlone` (binary indicator)
- **Result:** **81.56%** accuracy (+0.55% improvement) ⭐ **BEST RESULT**

### Attempt 2: Recursive Feature Elimination (RFE)
- **Method:** Selected top 5 features using RFE
- **Result:** 81.01% accuracy (no improvement, same as tuned model)



## Key Findings

1. **Random Forest consistently outperformed** other models across all experiments
2. **Hyperparameter tuning provided modest gains** (~0.5-2% depending on model)
3. **Feature engineering (FamilySize/IsAlone) was most effective**, achieving the highest accuracy at 81.56%
4. **Feature selection (RFE) didn't improve performance**, suggesting all features were valuable
5. **Logistic Regression showed no improvement** from hyperparameter tuning, indicating the default parameters were already optimal


## Final Recommendation

**Best Model Configuration:**
- **Algorithm:** Random Forest
- **Hyperparameters:** criterion='gini', max_depth=5, min_samples_split=10, n_estimators=10
- **Features:** Include FamilySize and IsAlone engineered features
- **Expected Accuracy:** **81.56%** on test set

**Classification Performance (Baseline Logistic Regression):**
- Class 0 (Did not survive): 81% precision, 85% recall
- Class 1 (Survived): 77% precision, 72% recall
- The model performs slightly better at predicting non-survivors