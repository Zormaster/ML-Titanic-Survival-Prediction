### Step 1
import pandas as pd
df = pd.read_csv('train.csv')

### Step 2
# Calculate the median age
median_age = df['Age'].median()
print(f"Median Age: {median_age}")

# Fill missing values with the median age
df['Age'] = df['Age'].fillna(median_age)

# Fill missing embarked values with the mode
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)
print(f"Embarked Mode: {embarked_mode}")

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)

### Step 3
# Split the data into features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

### Step 4
# Make predictions
y_pred = model.predict(X_test)

#Evaluate performance
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

### Step 5
# Compare different algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy:.4f}")

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    # 'Support Vector Machine': SVC(kernel='linear', random_state=42)
}
best_result = {'name': None, 'accuracy': 0, 'params': None}
for name, model in models.items():
    if name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    elif name == 'Decision Tree':
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
    elif name == 'Random Forest':
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy', 'log_loss']

        }
    # elif name == 'Support Vector Machine':
    #     param_grid = {
    #         'C': [0.1, 1, 10],
    #         'kernel': ['linear', 'rbf'],
    #         'gamma': ['scale', 'auto'],
    #         'decision_function_shape': ['ovo', 'ovr']
    #     }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name} after Hyperparameter Tuning: {accuracy:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    if accuracy > best_result['accuracy']:
        best_result['name'] = name
        best_result['accuracy'] = accuracy
        best_result['params'] = grid_search.best_params_

print("\nBest Overall Model:", best_result)

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_split=10, n_estimators=10, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Feature Engineering (Added FamilySize and IsAlone): {accuracy}")

# Advanced Feature Engineering
from sklearn.feature_selection import RFE
model = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_split=10, n_estimators=10, random_state=42)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)
model.fit(X_train_rfe, y_train)
y_pred = model.predict(X_test_rfe)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Feature Engineering: {accuracy}")

# Save the model
# import pickle
# with open('titanic_model.pkl', 'wb') as f:
#     pickle.dump(best_model, f)