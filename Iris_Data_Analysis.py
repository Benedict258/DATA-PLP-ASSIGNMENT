# Iris Data Analysis Assignment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Optional: Set seaborn style
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    print("✅ Dataset loaded successfully.\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Structure and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
print("\nDescriptive Statistics:")
print(df.describe())

# Group by species and compute mean
grouped = df.groupby('species').mean()
print("\nMean of each feature by species:")
print(grouped)

# Task 3: Visualizations

# 1. Line Chart - sepal length over index
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title("Line Chart: Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/line_chart_sepal_length.png")
plt.show()

# 2. Bar Chart - average petal length per species
plt.figure(figsize=(6, 4))
grouped['petal length (cm)'].plot(kind='bar', color='green')
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("/mnt/data/bar_chart_petal_length.png")
plt.show()

# 3. Histogram - distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=10, color='orange', edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("/mnt/data/histogram_sepal_width.png")
plt.show()

# 4. Scatter Plot - sepal length vs petal length, color by species
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set2', edgecolor='w')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("/mnt/data/scatter_sepal_vs_petal.png")
plt.show()
