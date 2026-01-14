import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global style
sns.set_theme(style="whitegrid")

def load_data(filepath):
    """Safely loads CSV data from the given filepath."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    return pd.read_csv(filepath)

def initial_exploration(df):
    """
    Prints basic statistics and information about the dataframe.
    - Head
    - Info (Data types and non-null counts)
    - Describe (Statistical summary of numerical columns)
    - Missing Value Counts
    - Duplicate ID Check
    """
    print("--- Head ---")
    print(df.head())
    print("\n--- Info ---")
    print(df.info())
    print("\n--- Describe ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicates ---")
    print(df.duplicated().sum())


def plot_distributions(df, output_dir):
    """
    Generates and saves exploratory visualization plots.
    1. Target Variable Distribution
    2. Numerical Feature Histograms
    3. Correlation Heatmap
    4. Categorical vs Target plots
    5. Numerical vs Target Boxplots
    """
    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Exited', data=df)
    plt.title('Distribution of Target (Exited)')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()

    # Numerical features
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[num_cols].hist(bins=20, figsize=(14, 10))
    plt.suptitle('Histograms of Numerical Features')
    plt.savefig(os.path.join(output_dir, 'numerical_histograms.png'))
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # Categorical vs Target
    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, hue='Exited', data=df, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'{col} vs Exited')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_vs_target.png'))
    plt.close()

    # Boxplots for Age/Balance/Salary by Target
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(x='Exited', y='Age', data=df, ax=axes[0])
    sns.boxplot(x='Exited', y='Balance', data=df, ax=axes[1])
    sns.boxplot(x='Exited', y='EstimatedSalary', data=df, ax=axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_vs_target_boxplots.png'))
    plt.close()

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Churn_Modelling.csv')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots', 'eda')
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(file_path)
    if df is not None:
        initial_exploration(df)
        print("Generating plots...")
        plot_distributions(df, output_dir)
        print(f"Plots saved to {output_dir}")

