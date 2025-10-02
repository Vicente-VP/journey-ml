"""
preprocessing_interests.py - Prepare data for course recommendation using profile data only
Focuses on interests as key feature using MultiLabelBinarizer
"""
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

# Type alias for clarity
DF = pd.DataFrame

# Configuration
PROFILES_PATH = "datasets/bv-profiles.csv"
OUTPUT_PATH = "datasets/course_recommendation_interests.csv"

ENCODABLE_COLS = [
    "experience",
    "education",
]

UNUSED_COLS = ["birthdate", "email", "state", "mbti"]


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def calculate_age(birthdate_series):
    """Calculate age from birthdate string (format: dd/mm/yyyy)"""
    today = datetime.today()
    ages = pd.to_datetime(birthdate_series, format="%d/%m/%Y").apply(
        lambda b: today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    )
    return ages


def explode_purchases(df):
    """
    Explode purchases so each row has one course target.
    Users with multiple purchases will have multiple rows.
    """
    # Filter out rows with no purchases
    df = df[df["purchases"].notna()].copy()
    
    # Split purchases by '|' and explode
    df["target_course"] = df["purchases"].str.split("|")
    df_exploded = df.explode("target_course")
    
    # Drop original purchases column
    df_exploded = df_exploded.drop(columns=["purchases"])
    
    # Reset index to avoid duplicate indices
    df_exploded = df_exploded.reset_index(drop=True)
    
    return df_exploded


def encode_interests(df):
    """
    Use MultiLabelBinarizer to split and encode interests field.
    interests is a '|' separated list like "Technology|Music|Sports"
    """
    # Split interests by '|' into lists
    df["interests_list"] = df["interests"].str.split("|")
    
    # Use MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    interests_encoded = mlb.fit_transform(df["interests_list"])
    
    # Create DataFrame with interest columns
    interests_df = pd.DataFrame(
        interests_encoded,
        columns=[f"interest_{interest}" for interest in mlb.classes_],
        index=df.index
    )
    
    print(f"  Found {len(mlb.classes_)} unique interests: {list(mlb.classes_)}")
    
    # Concatenate to original dataframe
    df = pd.concat([df, interests_df], axis=1)
    
    # Drop original interests columns
    df = df.drop(columns=["interests", "interests_list"])
    
    return df, mlb


def encode_categorical_features(df, encodable_cols):
    """One-hot encode categorical features"""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[encodable_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(encodable_cols),
        index=df.index
    )
    
    # Concatenate encoded features
    df = pd.concat([df, encoded_df], axis=1)
    
    # Drop original categorical columns
    df = df.drop(columns=encodable_cols)
    
    return df, encoder


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_data():
    """Main preprocessing workflow"""
    
    print("=" * 70)
    print("COURSE RECOMMENDATION DATA PREPROCESSING (PROFILE + INTERESTS ONLY)")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/6] Loading profiles data...")
    df = pd.read_csv(PROFILES_PATH)
    print(f"  Profiles: {len(df)} rows")
    
    # 2. Calculate age
    print("\n[2/6] Engineering features...")
    df["age"] = calculate_age(df["birthdate"])
    print(f"  Calculated age (range: {df['age'].min()}-{df['age'].max()})")
    
    # 3. Encode interests using MultiLabelBinarizer
    print("\n[3/6] Encoding interests with MultiLabelBinarizer...")
    df, mlb = encode_interests(df)
    
    # 4. Explode purchases
    print("\n[4/6] Exploding purchases into individual target courses...")
    df = explode_purchases(df)
    print(f"  Exploded to {len(df)} training examples")
    print(f"  Target courses: {df['target_course'].unique()}")
    print(f"  Course distribution:\n{df['target_course'].value_counts()}")
    
    # 5. Drop unused columns before encoding
    print("\n[5/6] Dropping unused columns...")
    df = df.drop(columns=UNUSED_COLS)
    
    # 6. Encode remaining categorical features
    print("\n[6/6] One-hot encoding remaining categorical features...")
    df, encoder = encode_categorical_features(df, ENCODABLE_COLS)
    print(f"  Final features: {len(df.columns)} columns")
    
    # Save
    print(f"\n{'=' * 70}")
    print(f"Saving to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Preprocessing complete!")
    print(f"  Final dataset shape: {df.shape}")
    print(f"  Target column: 'target_course'")
    print(f"{'=' * 70}\n")
    
    return df, encoder, mlb


# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    df_processed, encoder, mlb = preprocess_data()
    
    # Display sample
    print("\nSample of processed data:")
    print(df_processed.head())
    
    # Show interest columns
    interest_cols = [col for col in df_processed.columns if col.startswith('interest_')]
    print(f"\nInterest columns ({len(interest_cols)}):")
    print(interest_cols)
    
    print("\nAll columns:")
    print(df_processed.columns.tolist())