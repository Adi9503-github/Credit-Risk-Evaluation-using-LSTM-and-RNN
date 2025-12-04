# preprocessing utilities
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the dataset from CSV file."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Handle missing values using IterativeImputer and KNNImputer."""
    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Perform Iterative Imputation on numerical columns
    iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
    df[numerical_columns] = iterative_imputer.fit_transform(df[numerical_columns])

    # Perform KNN Imputation on the entire dataset
    knn_imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

    return df

def create_credit_risk_column(df, threshold_credit_amount=5000, threshold_duration=24):
    """Create credit_risk column based on thresholds."""
    df['credit_risk'] = np.where((df['Credit amount'] > threshold_credit_amount) & (df['Duration'] > threshold_duration), 1, 0)
    return df

def calculate_woe(df, column, target_column):
    """Calculate Weight of Evidence (WOE) for a categorical column."""
    df_temp = df[[column, target_column]].copy()
    df_temp['good'] = (df_temp[target_column] == 0).astype(int)
    df_temp['bad'] = (df_temp[target_column] == 1).astype(int)

    total_good = df_temp['good'].sum()
    total_bad = df_temp['bad'].sum()

    grouped = df_temp.groupby(column)
    woe = pd.Series()

    for category, group in grouped:
        good = group['good'].sum()
        bad = group['bad'].sum()

        if good == 0:
            good_percentage = 0.5  # Adjust for log(0)
        else:
            good_percentage = good / total_good

        if bad == 0:
            bad_percentage = 0.5  # Adjust for log(0)
        else:
            bad_percentage = bad / total_bad

        woe_value = np.log(good_percentage / bad_percentage)
        woe[category] = woe_value

    return woe

def apply_woe_encoding(df, target_column='credit_risk'):
    """Apply WOE encoding to categorical columns."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    woe_encodings = {}

    for column in categorical_columns:
        woe = calculate_woe(df, column, target_column)
        woe_encodings[column] = woe

    # Map WOE values to the corresponding categories in the dataset
    for column, woe in woe_encodings.items():
        df[column] = df[column].map(woe)

    return df

def scale_numerical_features(df, target_column='credit_risk'):
    """Scale numerical features using StandardScaler."""
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_columns_without_target = numerical_columns.drop(target_column, errors='ignore')

    scaler = StandardScaler()
    df[numerical_columns_without_target] = scaler.fit_transform(df[numerical_columns_without_target])

    return df, scaler

def preprocess_data(file_path):
    """Complete preprocessing pipeline."""
    # Load data
    df = load_data(file_path)

    # Handle missing values
    df = handle_missing_values(df)

    # Create credit_risk column
    df = create_credit_risk_column(df)

    # Apply WOE encoding
    df = apply_woe_encoding(df)

    # Scale numerical features
    df, scaler = scale_numerical_features(df)

    return df, scaler
