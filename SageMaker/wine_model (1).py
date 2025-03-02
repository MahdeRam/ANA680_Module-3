from __future__ import print_function
import os
import sys

# Ensure required packages are installed
os.system(f"{sys.executable} -m pip install joblib pandas numpy scikit-learn")

import joblib
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Set default directories for local execution
    default_output_dir = "/tmp/output"  # Local output directory
    default_model_dir = "/tmp/model"    # Local model directory
    default_train_dir = "/tmp/data"     # Local training data directory

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', default_output_dir))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', default_model_dir))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', default_train_dir))

    args = parser.parse_args()

    # Ensure model and data directories exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.train, exist_ok=True)

    # Check if dataset exists
    dataset_path = os.path.join(args.train, "wine_quality.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load dataset
    dataset = pd.read_csv(dataset_path)

    # Define features (X) and target variable (y)
    X = dataset.drop(columns=['quality'])
    y = dataset['quality']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model in SageMaker's expected location (or local fallback)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"Model training complete. Saved at {model_path}")
