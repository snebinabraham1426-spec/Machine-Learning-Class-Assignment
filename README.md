# Machine-Learning-Class-Assignment
# Loan Default Prediction using a Decision Tree Classifier **Machine Learning – Class Assignment**  This notebook follows the 12 steps in the assignment. Keep 'loan_default.csv' in the same folder as this notebook, or upload it to the current Google Colab session.
## Step 1: Import libraries and load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("loan_default.csv")
print("Dataset shape:", df.shape)
print("\\nColumn data types:")
df.info()
print("\\nFirst 10 rows:")
display(df.head(10))

## Step 2: Understand the data
print("Numeric summary statistics:")
display(df.describe())

categorical_cols = df.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    print(f"\\nValue counts for {col}:")
    display(df[col].value_counts(dropna=False))

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="loan_default", order=df["loan_default"].value_counts().index)
plt.title("Loan Default Class Distribution")
plt.xlabel("Loan default")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.show()

print(df["loan_default"].value_counts())
print((df["loan_default"].value_counts(normalize=True) * 100).round(2))

## Step 3: Clean the data
print("Missing values before cleaning:")
display(df.isnull().sum())

print("Exact duplicate rows:", df.duplicated().sum())
print("Negative values in numeric columns:")
numeric_original = df.select_dtypes(include=np.number)
display((numeric_original < 0).sum())

# Remove exact duplicate rows.
df = df.drop_duplicates().copy()

# Negative age or income is impossible; mark such values missing, then impute with the median.
for col in ["age", "income"]:
    df.loc[df[col] < 0, col] = np.nan

# Fill missing categorical values with the most frequent value.
for col in df.select_dtypes(include=["object", "category"]).columns:
    if df[col].isnull().any() and col != "customer_id":
        df[col] = df[col].fillna(df[col].mode()[0])

# Impute any missing numeric values with the median.
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

print("Shape after dropping exact duplicates:", df.shape)
print("Missing values after cleaning:")
display(df.isnull().sum())

## Step 4: Explore the data visually
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="loan_default", y="credit_score")
plt.title("Credit Score by Loan Default Status")
plt.tight_layout()
plt.show()

city_default = df.assign(default_flag=(df["loan_default"] == "Yes").astype(int)).groupby("city_tier")["default_flag"].mean().sort_index()
employment_default = df.assign(default_flag=(df["loan_default"] == "Yes").astype(int)).groupby("employment_type")["default_flag"].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
city_default.plot(kind="bar", ax=axes[0])
axes[0].set_title("Default Rate by City Tier")
axes[0].set_ylabel("Default rate")
axes[0].tick_params(axis="x", rotation=0)
employment_default.plot(kind="bar", ax=axes[1])
axes[1].set_title("Default Rate by Employment Type")
axes[1].set_ylabel("Default rate")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

numeric_for_corr = df.select_dtypes(include=np.number).copy()
numeric_for_corr["loan_default_binary"] = df["loan_default"].map({"No": 0, "Yes": 1})
plt.figure(figsize=(11, 8))
sns.heatmap(numeric_for_corr.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

print("Default rate by city tier:")
display((city_default * 100).round(2).rename("Default rate (%)"))
print("Default rate by employment type:")
display((employment_default * 100).round(2).rename("Default rate (%)"))

## Step 5: Prepare the data for modelling
model_df = df.drop(columns=["customer_id"]).copy()
model_df["loan_default"] = model_df["loan_default"].map({"Yes": 1, "No": 0})
model_df = pd.get_dummies(model_df, drop_first=True, dtype=int)

X = model_df.drop(columns=["loan_default"])
y = model_df["loan_default"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Encoded feature names:")
print(list(X.columns))

## Step 6: Split into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Training features:", X_train.shape)
print("Test features:", X_test.shape)
print("Training target:", y_train.shape)
print("Test target:", y_test.shape)

## Step 7: Train a Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
train_accuracy = dt.score(X_train, y_train)
test_accuracy = dt.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Test accuracy:     {test_accuracy:.3f}")

## Step 8: Evaluate the model properly
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
)

test_pred = dt.predict(X_test)
test_prob = dt.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No default", "Default"]).plot(cmap="Blues")
plt.title("Confusion Matrix – Untuned Decision Tree")
plt.tight_layout()
plt.show()

print(classification_report(y_test, test_pred, target_names=["No default", "Default"], zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.3f}")

## Step 9: Control overfitting by tuning
depth_results = []
for depth in [3, 5, 7, 10]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    depth_results.append({
        "max_depth": depth,
        "training_accuracy": model.score(X_train, y_train),
        "test_accuracy": model.score(X_test, y_test)
    })

depth_results_df = pd.DataFrame(depth_results)
display(depth_results_df.round(3))

from sklearn.model_selection import GridSearchCV
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"]
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print(f"Best mean cross-validation F1: {grid.best_score_:.3f}")
print(f"Best tree training accuracy: {best.score(X_train, y_train):.3f}")
print(f"Best tree test accuracy: {best.score(X_test, y_test):.3f}")
best_pred = best.predict(X_test)
best_prob = best.predict_proba(X_test)[:, 1]
print(classification_report(y_test, best_pred, target_names=["No default", "Default"], zero_division=0))
print(f"Best tree ROC-AUC: {roc_auc_score(y_test, best_prob):.3f}")

## Step 10: Compare with a baseline model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

log_reg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)
log_reg.fit(X_train, y_train)

def evaluate_model(name, model):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Recall (default)": recall_score(y_test, pred, zero_division=0),
        "F1 (default)": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }

comparison = pd.DataFrame([
    evaluate_model("Logistic Regression", log_reg),
    evaluate_model("Untuned Decision Tree", dt),
    evaluate_model("Tuned Decision Tree", best)
])
display(comparison.round(3))

## Step 11: Interpret the model
from sklearn.tree import plot_tree, export_text

importances = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 feature importances:")
display(importances.head(10).rename("Importance").to_frame())

plt.figure(figsize=(9, 5))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances – Tuned Decision Tree")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(22, 10))
plot_tree(
    best, feature_names=list(X.columns),
    class_names=["No default", "Default"],
    max_depth=3, filled=True, rounded=True, fontsize=9
)
plt.title("Tuned Decision Tree (first 3 levels)")
plt.tight_layout()
plt.show()

print(export_text(best, feature_names=list(X.columns), max_depth=3))

## Step 12: Conclusion
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
rf_comparison = pd.DataFrame([
    evaluate_model("Tuned Decision Tree", best),
    evaluate_model("Random Forest", rf)
])
display(rf_comparison.round(3))
