# preprocessing.py
# ============================================================
# Data preprocessing module for Personalized Learning Recommender
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. CLEANING FUNCTION
# ------------------------------------------------------------
def clean_dataset(df):
    """
    Basic cleaning: remove duplicates, handle missing values
    """
    print("\n=== CLEANING DATASET ===")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")

    # Handle missing values: simple strategy (can be improved)
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    df = df.fillna({
        col: df[col].mode()[0] if df[col].dtype == 'object' else df[col].median()
        for col in df.columns
    })
    print("Missing values filled (mode for categorical, median for numeric)")

    return df

# ------------------------------------------------------------
# 2. ENCODING FUNCTION
# ------------------------------------------------------------
def encode_features(df, categorical_cols, numerical_cols):
    """
    Encode categorical with LabelEncoder, scale numeric with MinMaxScaler
    """
    print("\n=== ENCODING FEATURES ===")

    # Label encode categorical
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded categorical: {col}")

    # Normalize numeric
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"Normalized numeric columns: {numerical_cols}")

    return df, encoders, scaler

# ------------------------------------------------------------
# 3. SPLIT FUNCTION
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split

def split_dataset(df, user_col, test_size=0.2, val_size=0.1, random_state=42, stratify=True, min_interactions=2):
    """
    Split dataset into train, val, test.
    
    Params:
    - df: dataframe
    - user_col: tên cột user
    - test_size: tỉ lệ test
    - val_size: tỉ lệ validation
    - stratify: nếu True thì stratify theo user_col (chỉ khi đủ dữ liệu)
    - min_interactions: số interactions tối thiểu để stratify
    
    Return: train, val, test
    """
    if stratify:
        # Lọc user có >= min_interactions
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df_strat = df[df[user_col].isin(valid_users)]

        if len(df_strat) > 0:
            print(f"Using stratify split on {len(valid_users)} users (>= {min_interactions} interactions)")
            train_val, test = train_test_split(
                df_strat,
                test_size=test_size,
                stratify=df_strat[user_col],
                random_state=random_state
            )
            train, val = train_test_split(
                train_val,
                test_size=val_size,
                stratify=train_val[user_col],
                random_state=random_state
            )
        else:
            print("⚠ Not enough users for stratify. Falling back to random split.")
            stratify = False

    if not stratify:
        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size, random_state=random_state)

    return train, val, test


# ------------------------------------------------------------
# 4. MASTER PIPELINE
# ------------------------------------------------------------
def preprocess_data(df, user_col, item_col, rating_col=None):
    """
    Complete preprocessing pipeline:
    1. Clean dataset
    2. Encode categorical + normalize numerical
    3. Split train/val/test
    """
    df = clean_dataset(df)

    # Auto-detect categorical & numeric
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove key IDs from normalization
    numerical_cols = [c for c in numerical_cols if c not in [user_col, item_col]]

    df, encoders, scaler = encode_features(df, categorical_cols, numerical_cols)

    train, val, test = split_dataset(
    df,
    user_col=user_col,
    stratify=True,      # bật stratify
    min_interactions=2  # chỉ stratify khi user có >= 2 interactions
)

    return train, val, test, encoders, scaler

