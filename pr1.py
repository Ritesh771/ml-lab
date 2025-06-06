import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

plt.figure(figsize=(12, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Histogram of {column}")
    plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[column])
    plt.title(f"Box Plot of {column}")
    plt.tight_layout()
plt.show()