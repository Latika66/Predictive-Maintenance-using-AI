import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath="data/ai4i2020.csv"):
    """Loads the dataset."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run download_data.py first.")
        return None

def clean_data(df):
    """Cleans the dataset by dropping ID columns and checking nulls."""
    df = df.copy()
    
    # Check nulls (should be 0 in this dataset, but good practice)
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        
    # Drop UDI and Product ID
    cols_to_drop = ["UDI", "Product ID"]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df

def feature_engineering(df):
    """Creates derived features and encodes categorical variables."""
    df = df.copy()
    
    # Derived Features
    # Power = Torque * Rotational speed
    # (Technically Power = Torque * Speed * 2pi / 60, but direct multiplication captures the interaction)
    # NOTE: the actual dataset (from ucimlrepo) has NO unit suffixes in column names
    # (e.g. "Torque", not "Torque [Nm]"). Matching on the raw names is required or
    # this feature silently never gets created.
    if "Torque" in df.columns and "Rotational speed" in df.columns:
        df["Power"] = df["Torque"] * df["Rotational speed"]
        
    # Temperature difference
    if "Process temperature" in df.columns and "Air temperature" in df.columns:
        df["Temp_Diff"] = df["Process temperature"] - df["Air temperature"]
        
    # One-Hot Encode 'Type' (L, M, H)
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=False)
        
    return df

def preprocess_pipeline(filepath="data/ai4i2020.csv", test_size=0.2, random_state=42):
    """Runs the full preprocessing pipeline and splits data."""
    df = load_data(filepath)
    if df is None:
        return None
        
    df = clean_data(df)
    df = feature_engineering(df)
    
    # Define features and target
    # The target is 'Machine failure'. 
    # Note: 'TWF', 'HDF', 'PWF', 'OSF', 'RNF' are sub-types of failure. 
    # For prediction, we should probably exclude them from training features as they are basically the target.
    target = "Machine failure"
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    X = df.drop(columns=[target] + failure_types, errors='ignore')
    y = df[target]
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numeric features
    scaler = StandardScaler()
    # We shouldn't scale the boolean one-hot encoded columns ideally, but scaling all is fine for these models.
    # To be precise, let's scale only float/int columns that are not one-hot encoded (or just scale all, it's standard)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names (important for SHAP)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns.tolist()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train class distribution:\n{y_train.value_counts(normalize=True)}")
