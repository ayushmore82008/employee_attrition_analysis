# ---------- train_model.py ----------
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("EmployeeAttrition_dataset.csv")
#It is an IBM HR Analytics Employee Attrition & Performance Dataset

# Select important features
selected_features = [
    'Age', 'MonthlyIncome', 'JobSatisfaction',
    'WorkLifeBalance', 'YearsAtCompany', 'OverTime'
]
df = df[selected_features + ['Attrition']]

# Encode categorical columns
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop missing values
df = df.dropna()

# Split Features and Target
X = df[selected_features]
y = df['Attrition']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Model 1: Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
print(f"Logistic Regression Accuracy: {log_acc * 100:.2f}%")

# Train Model 2: Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)
print(f"Decision Tree Accuracy: {tree_acc * 100:.2f}%")

# Train Model 3: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")

# Compare and Select Best Model
best_model = None
best_accuracy = 0
best_name = ""

if log_acc > best_accuracy:
    best_model = log_model
    best_accuracy = log_acc
    best_name = "Logistic Regression"

if tree_acc > best_accuracy:
    best_model = tree_model
    best_accuracy = tree_acc
    best_name = "Decision Tree"

if rf_acc > best_accuracy:
    best_model = rf_model
    best_accuracy = rf_acc
    best_name = "Random Forest"

print(f"\nBest Model: {best_name} ({best_accuracy * 100:.2f}% accuracy)")

# Save Best Model and Scaler
with open("attrition_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully!")
