import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("breast-cancer-cleaned.csv")

# Debugging Step: Print column names
print("Columns in dataset:", df.columns.tolist())

# Drop 'Sample code number' if it exists
if 'Sample code number' in df.columns:
    df.drop(columns=['Sample code number'], inplace=True)

# Convert target variable (Class: 2 → Benign, 4 → Malignant)
df['Class'] = df['Class'].map({2: 0, 4: 1})  # 0 = Benign, 1 = Malignant

# Define features and target
X = df.drop(columns=['Class'])  # Features (9 columns)
y = df['Class']  # Target

# Ensure dataset has the expected number of features
print(f"Total features used: {X.shape[1]}")  # Should be 9

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "breast_cancer_model.pkl")

print("Model training complete. Saved as 'breast_cancer_model.pkl'")




