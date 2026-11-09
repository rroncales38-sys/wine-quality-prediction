import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Kunin at suriin ang datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

print("=== DATASET INSPECTION ===")
print(f"Rows & Columns: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Quality distribution:\n{df['quality'].value_counts().sort_index()}")

df.to_csv("wine_quality.csv", index=False)

# Ihanda ang datos
df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)  # 1=Good, 0=Needs Improvement
X = df.drop(columns=["quality"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Bumuo at sanayin ang modelo
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Mga sukatan ng pagganap
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n=== MODEL EVALUATION ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(
    y_test, predictions, target_names=["Needs Improvement", "Good Quality"]
))

joblib.dump(model, "wine_model.pkl")
print("\n✅ Model saved as wine_model.pkl")
