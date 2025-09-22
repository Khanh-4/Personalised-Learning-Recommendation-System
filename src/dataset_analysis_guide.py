# Dataset Analysis Guide for Personalized Learning Recommendation System
# ========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 1. DATASET LOADING AND INITIAL EXPLORATION
# ========================================================================

def load_and_explore_dataset(file_path):
    """
    Load dataset and perform initial exploration
    """
    print("=== DATASET LOADING ===")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== BASIC INFO ===")
    print(df.info())
    
    print("\n=== FIRST 5 ROWS ===")
    print(df.head())
    
    print("\n=== COLUMN NAMES ===")
    print("Columns:", list(df.columns))
    
    return df

# ========================================================================
# 2. EXPECTED DATASET STRUCTURE ANALYSIS
# ========================================================================

def analyze_expected_structure(df):
    """
    Analyze if dataset contains expected columns for recommendation system
    """
    print("\n=== EXPECTED STRUCTURE ANALYSIS ===")
    
    # Expected columns for learning recommendation system
    expected_columns = {
        'user_columns': ['user_id', 'student_id', 'learner_id'],
        'item_columns': ['item_id', 'content_id', 'video_id', 'quiz_id', 'lesson_id'],
        'interaction_columns': ['rating', 'score', 'completion_rate', 'time_spent', 'clicked', 'viewed'],
        'temporal_columns': ['timestamp', 'date', 'time', 'created_at'],
        'context_columns': ['time_of_day', 'device', 'session_id'],
        'content_metadata': ['difficulty', 'topic', 'subject', 'type', 'duration'],
        'user_profile': ['age', 'gender', 'level', 'learning_style', 'performance']
    }
    
    found_columns = {}
    missing_categories = []
    
    for category, cols in expected_columns.items():
        found = [col for col in cols if any(col.lower() in df_col.lower() for df_col in df.columns)]
        actual_cols = [df_col for df_col in df.columns if any(col.lower() in df_col.lower() for col in cols)]
        
        found_columns[category] = actual_cols
        
        if actual_cols:
            print(f"✓ {category}: {actual_cols}")
        else:
            print(f"✗ {category}: Missing")
            missing_categories.append(category)
    
    return found_columns, missing_categories

# ========================================================================
# 3. DATA QUALITY ANALYSIS
# ========================================================================

def analyze_data_quality(df):
    """
    Comprehensive data quality analysis
    """
    print("\n=== DATA QUALITY ANALYSIS ===")
    
    # Missing values
    print("Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    # Unique values per column
    print("\nUnique Values per Column:")
    unique_counts = df.nunique().sort_values(ascending=False)
    print(unique_counts.head(10))
    
    return missing_df, duplicates, unique_counts

# ========================================================================
# 4. USER-ITEM INTERACTION ANALYSIS
# ========================================================================

def analyze_user_item_interactions(df, user_col, item_col, rating_col=None):
    """
    Analyze user-item interaction patterns
    """
    print(f"\n=== USER-ITEM INTERACTION ANALYSIS ===")
    
    # Basic statistics
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_interactions = len(df)
    
    print(f"Number of unique users: {n_users:,}")
    print(f"Number of unique items: {n_items:,}")
    print(f"Number of interactions: {n_interactions:,}")
    print(f"Sparsity: {(1 - n_interactions / (n_users * n_items)) * 100:.4f}%")
    
    # User interaction distribution
    user_interactions = df.groupby(user_col).size()
    print(f"\nUser Interaction Statistics:")
    print(f"Mean interactions per user: {user_interactions.mean():.2f}")
    print(f"Median interactions per user: {user_interactions.median():.2f}")
    print(f"Min interactions per user: {user_interactions.min()}")
    print(f"Max interactions per user: {user_interactions.max()}")
    
    # Item interaction distribution
    item_interactions = df.groupby(item_col).size()
    print(f"\nItem Interaction Statistics:")
    print(f"Mean interactions per item: {item_interactions.mean():.2f}")
    print(f"Median interactions per item: {item_interactions.median():.2f}")
    print(f"Min interactions per item: {item_interactions.min()}")
    print(f"Max interactions per item: {item_interactions.max()}")
    
    # Rating distribution (if available)
    if rating_col and rating_col in df.columns:
        print(f"\nRating Distribution:")
        print(df[rating_col].describe())
        print(f"Rating value counts:")
        print(df[rating_col].value_counts().sort_index())
    
    return {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': n_interactions,
        'user_interactions': user_interactions,
        'item_interactions': item_interactions
    }

# ========================================================================
# 5. TEMPORAL ANALYSIS
# ========================================================================

def analyze_temporal_patterns(df, timestamp_col):
    """
    Analyze temporal patterns in interactions
    """
    print(f"\n=== TEMPORAL ANALYSIS ===")
    
    if timestamp_col not in df.columns:
        print(f"Timestamp column '{timestamp_col}' not found")
        return None
    
    # Convert to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Basic temporal statistics
    print(f"Time range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
    print(f"Total duration: {(df[timestamp_col].max() - df[timestamp_col].min()).days} days")
    
    # Extract temporal features
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month
    
    # Hourly distribution
    print("\nHourly Distribution (top 5):")
    hourly_dist = df['hour'].value_counts().sort_index()
    print(hourly_dist.head())
    
    # Daily distribution
    print("\nDaily Distribution:")
    daily_dist = df['day_of_week'].value_counts().sort_index()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day, count in daily_dist.items():
        print(f"{day_names[day]}: {count}")
    
    return {
        'hourly_dist': hourly_dist,
        'daily_dist': daily_dist,
        'time_range': (df[timestamp_col].min(), df[timestamp_col].max())
    }

# ========================================================================
# 6. CONTENT ANALYSIS
# ========================================================================

def analyze_content_metadata(df, content_cols):
    """
    Analyze content metadata for recommendation features
    """
    print(f"\n=== CONTENT METADATA ANALYSIS ===")
    
    for col in content_cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            if df[col].dtype == 'object':
                # Categorical analysis
                value_counts = df[col].value_counts()
                print(f"Unique values: {df[col].nunique()}")
                print(f"Top 5 values:")
                print(value_counts.head())
            else:
                # Numerical analysis
                print(df[col].describe())
        else:
            print(f"Column '{col}' not found")

# ========================================================================
# 7. RECOMMENDATION SYSTEM SUITABILITY
# ========================================================================

def assess_recommendation_suitability(df, interaction_stats):
    """
    Assess if dataset is suitable for recommendation system
    """
    print(f"\n=== RECOMMENDATION SYSTEM SUITABILITY ===")
    
    # Sparsity check
    sparsity = 1 - (interaction_stats['n_interactions'] / 
                   (interaction_stats['n_users'] * interaction_stats['n_items']))
    
    suitability_score = 0
    issues = []
    recommendations = []
    
    # Check 1: Sparsity
    if sparsity < 0.99:
        print("✓ Sparsity acceptable (<99%)")
        suitability_score += 1
    else:
        print("⚠ Very high sparsity (>99%)")
        issues.append("High sparsity")
        recommendations.append("Consider cold start handling")
    
    # Check 2: User distribution
    user_median = interaction_stats['user_interactions'].median()
    if user_median >= 5:
        print("✓ Users have sufficient interactions")
        suitability_score += 1
    else:
        print("⚠ Users have few interactions")
        issues.append("Few interactions per user")
        recommendations.append("Implement content-based features")
    
    # Check 3: Item distribution
    item_median = interaction_stats['item_interactions'].median()
    if item_median >= 3:
        print("✓ Items have sufficient interactions")
        suitability_score += 1
    else:
        print("⚠ Items have few interactions")
        issues.append("Few interactions per item")
        recommendations.append("Use item metadata for cold start")
    
    # Check 4: Dataset size
    if interaction_stats['n_interactions'] >= 10000:
        print("✓ Sufficient number of interactions")
        suitability_score += 1
    else:
        print("⚠ Small dataset")
        issues.append("Small dataset")
        recommendations.append("Consider data augmentation")
    
    print(f"\nSuitability Score: {suitability_score}/4")
    
    if issues:
        print(f"Issues to address: {', '.join(issues)}")
        print(f"Recommendations: {', '.join(recommendations)}")
    
    return suitability_score, issues, recommendations

# ========================================================================
# 8. VISUALIZATION FUNCTIONS
# ========================================================================

def create_visualizations(df, user_col, item_col, interaction_stats):
    """
    Create visualizations for dataset analysis
    """
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # User interaction distribution
    user_interactions = interaction_stats['user_interactions']
    axes[0,0].hist(user_interactions, bins=50, alpha=0.7)
    axes[0,0].set_title('Distribution of User Interactions')
    axes[0,0].set_xlabel('Number of Interactions')
    axes[0,0].set_ylabel('Number of Users')
    axes[0,0].axvline(user_interactions.median(), color='red', linestyle='--', 
                      label=f'Median: {user_interactions.median():.1f}')
    axes[0,0].legend()
    
    # Item interaction distribution
    item_interactions = interaction_stats['item_interactions']
    axes[0,1].hist(item_interactions, bins=50, alpha=0.7)
    axes[0,1].set_title('Distribution of Item Interactions')
    axes[0,1].set_xlabel('Number of Interactions')
    axes[0,1].set_ylabel('Number of Items')
    axes[0,1].axvline(item_interactions.median(), color='red', linestyle='--',
                      label=f'Median: {item_interactions.median():.1f}')
    axes[0,1].legend()
    
    # Long tail analysis - Users
    user_sorted = user_interactions.sort_values(ascending=False)
    cumsum_users = user_sorted.cumsum() / user_sorted.sum()
    axes[1,0].plot(range(len(cumsum_users)), cumsum_users)
    axes[1,0].set_title('User Interaction Long Tail')
    axes[1,0].set_xlabel('User Rank')
    axes[1,0].set_ylabel('Cumulative Interaction Percentage')
    axes[1,0].grid(True, alpha=0.3)
    
    # Long tail analysis - Items
    item_sorted = item_interactions.sort_values(ascending=False)
    cumsum_items = item_sorted.cumsum() / item_sorted.sum()
    axes[1,1].plot(range(len(cumsum_items)), cumsum_items)
    axes[1,1].set_title('Item Interaction Long Tail')
    axes[1,1].set_xlabel('Item Rank')
    axes[1,1].set_ylabel('Cumulative Interaction Percentage')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ========================================================================
# 9. FEATURE ENGINEERING SUGGESTIONS
# ========================================================================

def suggest_feature_engineering(df, found_columns):
    """
    Suggest feature engineering based on available columns
    """
    print(f"\n=== FEATURE ENGINEERING SUGGESTIONS ===")
    
    suggestions = []
    
    # User features
    if found_columns.get('user_profile'):
        suggestions.append("✓ User demographic features available")
        suggestions.append("  - One-hot encode categorical user features")
        suggestions.append("  - Normalize numerical user features")
    
    # Temporal features
    if found_columns.get('temporal_columns'):
        suggestions.append("✓ Temporal features available")
        suggestions.append("  - Extract hour_of_day, day_of_week features")
        suggestions.append("  - Create recency features")
        suggestions.append("  - Calculate session-based features")
    
    # Content features
    if found_columns.get('content_metadata'):
        suggestions.append("✓ Content metadata available")
        suggestions.append("  - Create content embeddings")
        suggestions.append("  - One-hot encode content categories")
        suggestions.append("  - Calculate content popularity features")
    
    # Interaction features
    if found_columns.get('interaction_columns'):
        suggestions.append("✓ Interaction features available")
        suggestions.append("  - Calculate user/item average ratings")
        suggestions.append("  - Create interaction frequency features")
        suggestions.append("  - Compute engagement metrics")
    
    # Missing features suggestions
    missing_features = []
    if not found_columns.get('user_profile'):
        missing_features.append("Consider collecting user demographic data")
    if not found_columns.get('content_metadata'):
        missing_features.append("Add content metadata (difficulty, topic, type)")
    if not found_columns.get('temporal_columns'):
        missing_features.append("Include timestamp information")
    
    print("\nRecommended Features to Create:")
    for suggestion in suggestions:
        print(suggestion)
    
    if missing_features:
        print("\nMissing Feature Categories:")
        for missing in missing_features:
            print(f"- {missing}")
    
    return suggestions

# ========================================================================
# 10. MAIN ANALYSIS PIPELINE
# ========================================================================

def run_complete_analysis(file_path):
    """
    Run complete dataset analysis pipeline
    """
    print("STARTING COMPLETE DATASET ANALYSIS")
    print("=" * 50)
    
    try:
        # 1. Load and explore
        df = load_and_explore_dataset(file_path)
        
        # 2. Structure analysis
        found_columns, missing_categories = analyze_expected_structure(df)
        
        # 3. Data quality
        missing_df, duplicates, unique_counts = analyze_data_quality(df)
        
        # 4. Identify key columns (you may need to adjust these based on your dataset)
        # These are common column names - adjust based on your actual dataset
        potential_user_cols = [col for col in df.columns 
                              if any(term in col.lower() for term in ['user', 'student', 'learner'])]
        potential_item_cols = [col for col in df.columns 
                              if any(term in col.lower() for term in ['item', 'content', 'video', 'quiz', 'lesson'])]
        potential_rating_cols = [col for col in df.columns 
                                if any(term in col.lower() for term in ['rating', 'score', 'grade'])]
        potential_time_cols = [col for col in df.columns 
                              if any(term in col.lower() for term in ['time', 'date', 'timestamp'])]
        
        print(f"\nPotential key columns identified:")
        print(f"User columns: {potential_user_cols}")
        print(f"Item columns: {potential_item_cols}")
        print(f"Rating columns: {potential_rating_cols}")
        print(f"Time columns: {potential_time_cols}")
        
        # 5. User-item analysis (if key columns found)
        if potential_user_cols and potential_item_cols:
            user_col = potential_user_cols[0]
            item_col = potential_item_cols[0]
            rating_col = potential_rating_cols[0] if potential_rating_cols else None
            
            interaction_stats = analyze_user_item_interactions(df, user_col, item_col, rating_col)
            
            # 6. Temporal analysis (if time column found)
            if potential_time_cols:
                time_col = potential_time_cols[0]
                temporal_stats = analyze_temporal_patterns(df, time_col)
            
            # 7. Content analysis
            content_cols = found_columns.get('content_metadata', [])
            if content_cols:
                analyze_content_metadata(df, content_cols)
            
            # 8. Suitability assessment
            suitability_score, issues, recommendations = assess_recommendation_suitability(df, interaction_stats)
            
            # 9. Create visualizations
            create_visualizations(df, user_col, item_col, interaction_stats)
            
        else:
            print("\n⚠ Could not identify user and item columns automatically.")
            print("Please manually specify the column names for user_id and item_id.")
        
        # 10. Feature engineering suggestions
        suggestions = suggest_feature_engineering(df, found_columns)
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        
        return df, found_columns, interaction_stats if 'interaction_stats' in locals() else None
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None, None, None

# ========================================================================
# 11. USAGE EXAMPLE
# ========================================================================

if __name__ == "__main__":
    # Example usage - replace with your actual file path
    file_path = "../dataset/personalized_learning_dataset.csv"  # Replace with actual path
    
    # Run analysis
    df, found_columns, interaction_stats = run_complete_analysis(file_path)
    
    # Manual column specification example (if auto-detection fails)
    """
    # If auto-detection fails, manually specify columns:
    user_col = 'user_id'  # Replace with actual column name
    item_col = 'item_id'  # Replace with actual column name
    rating_col = 'rating'  # Replace with actual column name
    timestamp_col = 'timestamp'  # Replace with actual column name
    
    interaction_stats = analyze_user_item_interactions(df, user_col, item_col, rating_col)
    temporal_stats = analyze_temporal_patterns(df, timestamp_col)
    """

# ========================================================================
# 12. NEXT STEPS CHECKLIST
# ========================================================================

def print_next_steps():
    """
    Print next steps after dataset analysis
    """
    print("\n=== NEXT STEPS AFTER DATASET ANALYSIS ===")
    next_steps = [
        "1. ✓ Complete dataset analysis",
        "2. ⏳ Data preprocessing and cleaning",
        "3. ⏳ Train-validation-test split",
        "4. ⏳ Implement baseline models (Popularity, MF)",
        "5. ⏳ Create evaluation framework",
        "6. ⏳ Implement advanced models (NCF, SASRec)",
        "7. ⏳ Add knowledge graph features",
        "8. ⏳ Implement contextual bandit",
        "9. ⏳ Build demo interface",
        "10. ⏳ Write final report"
    ]
    
    for step in next_steps:
        print(step)

print_next_steps()