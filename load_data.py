import pandas as pd

# Download dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# Save locally
df.to_csv("wine_quality.csv", index=False)
print("Dataset downloaded successfully!")
