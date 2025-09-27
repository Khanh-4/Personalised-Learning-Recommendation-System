# src/preprocessing.py
# ============================================================
# Data preprocessing pipeline for Personalized Learning System
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("=== CLEANING DATASET ===")
    before = df.shape[0]

    # remove duplicates
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows")

    # missing values
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    print("Missing values filled (mode for categorical, median for numeric)")
    return df


# ------------------------------------------------------------
# Encoding
# ------------------------------------------------------------
def encode_features(df: pd.DataFrame, categorical_cols, numeric_cols):
    print("\n=== ENCODING FEATURES ===")

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"Encoded categorical: {col}")

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"Normalized numeric columns: {numeric_cols}")

    return df, encoders, scaler


# ------------------------------------------------------------
# Time-based split per user
# ------------------------------------------------------------
def split_dataset_time_based(
    df, user_col="learner_id", time_col="timestamp",
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
):
    """
    Split dataset per user in chronological order.
    Guarantees each user has interactions in train set.
    """
    train, val, test = [], [], []

    for user, user_df in df.groupby(user_col):
        user_df = user_df.sort_values(time_col)
        n = len(user_df)

        if n < 3:
            train.append(user_df)
            continue

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train.append(user_df.iloc[:train_end])
        if val_end > train_end:
            val.append(user_df.iloc[train_end:val_end])
        if val_end < n:
            test.append(user_df.iloc[val_end:])

    train = pd.concat(train).reset_index(drop=True)
    val = pd.concat(val).reset_index(drop=True) if val else pd.DataFrame(columns=df.columns)
    test = pd.concat(test).reset_index(drop=True) if test else pd.DataFrame(columns=df.columns)

    print(f"[Split] Train={train.shape}, Val={val.shape}, Test={test.shape}")
    return train, val, test


# ------------------------------------------------------------
# Main preprocessing pipeline
# ------------------------------------------------------------
def preprocess_data(df, user_col="learner_id", item_col="content_type", rating_col="engagement_score"):
    # Clean
    df = clean_dataset(df)

    # Feature engineering
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    categorical_cols = [user_col, "session_id", item_col]
    numeric_cols = [
        "time_spent",
        rating_col,
        "quiz_score",
        "completion_rate",
        "attempts_per_quiz",
        "learning_outcome",
        "hour",
        "day_of_week",
        "month",
    ]

    # Encode
    df, encoders, scaler = encode_features(df, categorical_cols, numeric_cols)

    # Split
    train, val, test = split_dataset_time_based(df, user_col=user_col, time_col="timestamp")

    return train, val, test, encoders, scaler
