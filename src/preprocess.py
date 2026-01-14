import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import CatBoostEncoder

from category_encoders import CatBoostEncoder

# Define Feature Engineer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to generate new features from existing columns.
    
    New Features:
    - BalanceSalaryRatio: Ratio of Balance to EstimatedSalary.
    - TenureByAge: Ratio of Tenure to Age.
    - CreditScoreGivenAge: Ratio of CreditScore to Age.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ratios providing interaction between financial status and demographics
        X['BalanceSalaryRatio'] = X['Balance'] / X['EstimatedSalary']
        X['TenureByAge'] = X['Tenure'] / X['Age']
        X['CreditScoreGivenAge'] = X['CreditScore'] / X['Age']
        return X

def preprocess_data():
    """
    Main preprocessing function.
    
    Steps:
    1. Loads raw dataset.
    2. Drops irrelevant columns (RowNumber, CustomerId).
    3. Performs Stratified Split to maintain target distribution.
    4. Applies Feature Engineering.
    5. Encodes Categorical Variables:
       - One-Hot Encoding for low-cardinality nominal features (Geography, Gender).
       - CatBoost Encoding for high-cardinality nominal features (Surname).
    6. Saves processed datasets.
    """
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'Churn_Modelling.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Load Data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Drop irrelevant identifiers
    drop_cols = ['RowNumber', 'CustomerId']
    print(f"Dropping columns: {drop_cols}")
    df = df.drop(columns=drop_cols)
    
    # Separate Target
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Split Data (Stratified to handle class imbalance)
    print("Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define feature groups for encoding
    categorical_ohe = ['Geography', 'Gender']
    categorical_cbe = ['Surname']
    
    # 1. Feature Engineering
    print("Applying Feature Engineering...")
    fe = FeatureEngineer()
    X_train = fe.transform(X_train)
    X_test = fe.transform(X_test)
    
    # 2. Encoding
    print("Applying Encoders...")
    # One-Hot Encoder for low cardinality features
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    # CatBoost Encoder for high cardinality 'Surname'
    # CatBoostEncoder uses target statistics, so it is fitted on training data.
    cbe = CatBoostEncoder(cols=categorical_cbe, random_state=42)
    
    # Fit Encoders
    ohe.fit(X_train[categorical_ohe])
    cbe.fit(X_train[categorical_cbe], y_train)
    
    # Transform Helper
    def transform_part(X_part, y_part=None, is_train=False):
        """
        Applies fitted encoders to a dataset partition.
        Handles CatBoostEncoder's requirement for target usage during training
        to prevent data leakage (regularization).
        """
        # OHE Transform
        ohe_vals = ohe.transform(X_part[categorical_ohe])
        ohe_cols = ohe.get_feature_names_out(categorical_ohe)
        ohe_df = pd.DataFrame(ohe_vals, columns=ohe_cols, index=X_part.index)
        
        # CBE Transform
        if is_train and y_part is not None:
             # Using target for training set transformation introduces regularization
             cbe_vals = cbe.transform(X_part[categorical_cbe], y_part)
        else:
             # Test set is transformed using statistics learnt from training set
             cbe_vals = cbe.transform(X_part[categorical_cbe])
             
        cbe_df = pd.DataFrame(cbe_vals, columns=categorical_cbe, index=X_part.index).add_suffix('_CBE')

        # Combine: Numerical leftovers + OHE + CBE
        base_df = X_part.drop(columns=categorical_ohe + categorical_cbe)
        
        return pd.concat([base_df, ohe_df, cbe_df], axis=1)

    X_train_processed = transform_part(X_train, y_train, is_train=True)
    X_test_processed = transform_part(X_test, is_train=False)
    
    # Save outputs
    print("Saving processed files...")
    X_train_processed.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_processed.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print("Preprocessing Complete.")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")

if __name__ == "__main__":
    preprocess_data()
