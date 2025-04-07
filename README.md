# PYTHON-6

# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Iris dataset as an example)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(url, header=None, names=columns)

# Display the first few rows to inspect the data
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset (data types and missing values)
print("\nData types and missing values:")
print(df.info())

# Clean the dataset: Check for and handle missing values (None in this case)
# As there are no missing values, no need for imputation or removal here
# If there were missing values, we would use df.fillna() or df.dropna()

# Task 2: Basic Data Analysis

# Compute basic statistics of the numerical columns
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Perform groupings by 'species' and compute the mean of numerical columns
species_grouped = df.groupby('species').mean()
print("\nAverage values per species:")
print(species_grouped)

# Task 3: Data Visualization

# Line chart showing trends over time (this could be a time-series if there is a time column)
# For demonstration purposes, let's assume we're plotting one of the numerical columns over the index
plt.figure(figsize=(10,6))
df['sepal_length'].plot(kind='line', title='Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.show()

# Bar chart showing the comparison of numerical values (average petal length per species)
plt.figure(figsize=(10,6))
sns.barplot(x=species_grouped.index, y=species_grouped['petal_length'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')
plt.show()

# Histogram of a numerical column (petal length distribution)
plt.figure(figsize=(10,6))
df['petal_length'].hist(bins=20)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()

# Scatter plot to visualize the relationship between sepal length and petal length
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['sepal_length'], y=df['petal_length'], hue=df['species'])
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Species')
plt.show()

# Additional error handling example
try:
    # Trying to read a non-existent file for error handling demonstration
    df_non_existent = pd.read_csv('non_existent_file.csv')
except FileNotFoundError:
    print("\nError: The file was not found. Please check the file path and try again.")
