# File Path: dataset_analysis.py

# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Task 1: Loading and Exploring the Dataset
def load_and_explore_dataset():
    try:
        # Load the Iris dataset
        iris = load_iris(as_frame=True)
        df = iris['frame']

        # Display first few rows
        print("First few rows of the dataset:")
        print(df.head())

        # Check data types and missing values
        print("\nDataset Info:")
        print(df.info())

        print("\nChecking for missing values:")
        print(df.isnull().sum())

        # Clean dataset (Iris has no missing values, but this is an example)
        if df.isnull().values.any():
            df.fillna(df.mean(numeric_only=True), inplace=True)

        return df, iris
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

# Task 2: Basic Data Analysis
def basic_data_analysis(df):
    # Basic statistics
    print("\nBasic statistics of numerical columns:")
    print(df.describe())

    # Grouping and mean calculation
    print("\nMean sepal length by species:")
    grouped_data = df.groupby('target')['sepal length (cm)'].mean()
    print(grouped_data)

    return grouped_data

import numpy as np  # Ensure numpy is imported

def visualize_data(df, iris):
    # Setting up seaborn style
    sns.set_theme(style="whitegrid")

    # Line chart
    plt.figure(figsize=(10, 6))
    df['index'] = np.arange(1, len(df) + 1)  # Ensure numpy is imported before this line
    plt.plot(df['index'], df['sepal length (cm)'], label="Sepal Length", color='blue')
    plt.title("Sepal Length Trend")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.pause(0.1)
    print("Line plot displayed")

    # Bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(x=df['target'], y=df['petal length (cm)'], ci=None, palette='muted')
    plt.title("Average Petal Length by Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.xticks(ticks=[0, 1, 2], labels=iris['target_names'])
    plt.pause(0.1)
    print("Bar chart displayed")

    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df['petal length (cm)'], kde=True, color='green', bins=15)
    plt.title("Distribution of Petal Length")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Frequency")
    plt.pause(0.1)
    print("Histogram displayed")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['sepal length (cm)'], y=df['petal length (cm)'], hue=df['target'], palette='deep')
    plt.title("Sepal Length vs. Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title='Species', loc='upper left', labels=iris['target_names'])
    plt.pause(0.1)
    print("Scatter plot displayed")

# Main function to orchestrate the tasks
def main():
    df, iris = load_and_explore_dataset()
    if df is not None:
        basic_data_analysis(df)
        visualize_data(df, iris)

if __name__ == "__main__":
    main()
