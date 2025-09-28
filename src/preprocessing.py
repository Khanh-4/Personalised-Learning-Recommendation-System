# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Thêm hàm stratified split ---
def stratified_user_split(df, user_col, test_size=0.2, random_state=42):
    train_parts, test_parts = [], []
    for u, group in df.groupby(user_col):
        if len(group) == 1:
            # user chỉ có 1 record → cho hết vào train
            train_parts.append(group)
        else:
            tr, te = train_test_split(group, test_size=test_size, random_state=random_state)
            train_parts.append(tr)
            test_parts.append(te)
    return pd.concat(train_parts), pd.concat(test_parts)


def preprocess_data(
    df,
    user_col="learner_id",
    item_col="content_type",
    rating_col="engagement_score",
    test_size=0.2,
    val_size=0.1,
    random_state=42,
):
    """
    Preprocess dataframe:
      - drop duplicates
      - fill missing (mode for cat, median for num)
      - encode categorical (LabelEncoder)
      - SCALE: rating_col -> MinMaxScaler (0..1); other numeric columns -> StandardScaler
      - split train/val/test (theo user, tránh cold-start)
    Returns: train, val, test, encoders, scalers
    """
    print("=== CLEANING DATASET ===")
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Removed {before - len(df)} duplicate rows")

    # Fill missing
    total_missing = df.isnull().sum().sum()
    print(f"Total missing values: {total_missing}")
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
            else:
                df[col] = df[col].fillna(df[col].median())
    print("Missing values filled (mode for categorical, median for numeric)")

    # Encode categorical object columns
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded categorical: {col}")

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Loại trừ user_col và item_col khỏi scaling
    exclude_cols = [user_col, item_col]
    num_cols = [c for c in num_cols if c not in exclude_cols]

    rating_scaler = None
    standard_scaler = None

    if rating_col and rating_col in num_cols:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0.0)
        mm = MinMaxScaler(feature_range=(0, 1))
        df[[rating_col]] = mm.fit_transform(df[[rating_col]])
        rating_scaler = mm
        num_cols.remove(rating_col)
        print(f"Scaled rating column '{rating_col}' to [0,1] using MinMaxScaler")

    # Standard scale các numeric khác
    if num_cols:
        ss = StandardScaler()
        df[num_cols] = ss.fit_transform(df[num_cols])
        standard_scaler = ss
        print(f"Standard-scaled numeric columns (except ID, rating): {num_cols}")

    # --- THAY phần split ở đây ---
    train_full, test = stratified_user_split(df, user_col=user_col, test_size=test_size, random_state=random_state)
    train, val = stratified_user_split(train_full, user_col=user_col, test_size=val_size, random_state=random_state)

    print(f"Train shape: {train.shape}")
    print(f"Val shape:   {val.shape}")
    print(f"Test shape:  {test.shape}")

    scalers = {"standard_scaler": standard_scaler, "rating_scaler": rating_scaler}
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), encoders, scalers
