"""
Video Game Rating Classification Project
---------------------------------------

Dataset:
Video Game Sales with Ratings
https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings

Goal:
Predict whether a video game's critic rating is Low, Medium, or High
using game features such as platform, genre, publisher, sales, and user score.

Models used:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)

Before running:
1. Download the dataset from Kaggle.
2. Place "Video_Games_Sales_as_at_22_Dec_2016.csv" in the same folder as this file.
3. Run this file in Python, Jupyter, VS Code, or another Python environment.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# =========================
# 1. Load Dataset
# =========================

DATA_FILE = "Video_Games_Sales_as_at_22_Dec_2016.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# =========================
# 2. Basic Data Exploration
# =========================

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))


# =========================
# 3. Visual Analysis
# =========================

plt.figure(figsize=(8, 5))
plt.hist(df["Critic_Score"].dropna(), bins=20)
plt.xlabel("Critic Score")
plt.ylabel("Number of Games")
plt.title("Distribution of Critic Scores")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df["Genre"].value_counts().plot(kind="bar")
plt.xlabel("Genre")
plt.ylabel("Number of Games")
plt.title("Number of Games by Genre")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df["Platform"].value_counts().head(15).plot(kind="bar")
plt.xlabel("Platform")
plt.ylabel("Number of Games")
plt.title("Top 15 Platforms by Number of Games")
plt.tight_layout()
plt.show()


# =========================
# 4. Preprocessing
# =========================

features = [
    "Platform",
    "Genre",
    "Publisher",
    "Year_of_Release",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "User_Score",
    "User_Count",
    "Critic_Count"
]

target = "Critic_Score"

data = df[features + [target]].copy()

# User_Score sometimes contains text values like "tbd", so convert it to numeric.
data["User_Score"] = pd.to_numeric(data["User_Score"], errors="coerce")

# Drop rows with missing values for this starter version.
data = data.dropna()

def rating_class(score):
    """Convert critic score into a rating category."""
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

data["Rating_Class"] = data[target].apply(rating_class)

X = data[features]
y = data["Rating_Class"]

print("\nCleaned dataset shape:", data.shape)
print("\nRating class distribution:")
print(y.value_counts())


# =========================
# 5. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

categorical_features = ["Platform", "Genre", "Publisher"]

numeric_features = [
    "Year_of_Release",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "User_Score",
    "User_Count",
    "Critic_Count"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)


# =========================
# 6. Model 1: Logistic Regression
# =========================

log_reg_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

log_reg_model.fit(X_train, y_train)
log_preds = log_reg_model.predict(X_test)

print("\n==============================")
print("Logistic Regression Results")
print("==============================")
print("Accuracy:", accuracy_score(y_test, log_preds))
print(classification_report(y_test, log_preds))

ConfusionMatrixDisplay.from_predictions(y_test, log_preds)
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()


# =========================
# 7. Model 2: K-Nearest Neighbors
# =========================

knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]
)

knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

print("\n==============================")
print("K-Nearest Neighbors Results")
print("==============================")
print("Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds))

ConfusionMatrixDisplay.from_predictions(y_test, knn_preds)
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.show()


# =========================
# 8. Compare Model Results
# =========================

results = pd.DataFrame({
    "Model": ["Logistic Regression", "K-Nearest Neighbors"],
    "Accuracy": [
        accuracy_score(y_test, log_preds),
        accuracy_score(y_test, knn_preds)
    ]
})

print("\nModel Comparison:")
print(results)
