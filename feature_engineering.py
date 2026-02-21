"""
Feature Engineering Module for Loan Data Analysis

This module provides functions for creating derived features from loan data
to improve model performance.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

try:
    from . import text_util as tu
except ImportError:
    import text_util as tu


@dataclass
class PreparedData:
    """
    Container for preprocessed training data.

    Attributes:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features (scaled if scale_features=True)
    X_val : pd.DataFrame or np.ndarray or None
        Validation features (scaled if scale_features=True)
    X_test : pd.DataFrame or np.ndarray
        Test features (scaled if scale_features=True)
    y_train : pd.Series
        Training target
    y_val : pd.Series or None
        Validation target
    y_test : pd.Series
        Test target
    scaler : StandardScaler or None
        StandardScaler object (if scale_features=True, else None)
    encoded_df : pd.DataFrame
        DataFrame after encoding (before split)
    feature_names : List[str]
        List of feature column names
    """
    X_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: Optional[pd.Series]
    y_test: pd.Series
    scaler: Optional[object]
    encoded_df: pd.DataFrame
    feature_names: List[str]


def create_loan_features(df):
    """
    Create engineered features for loan default prediction.

    Creates the following features:
    - credit_util_ratio: Normalized revolving credit utilization
    - annual_inquiry_rate: Projected annual credit inquiries
    - debt_burden: Monthly payment relative to annual income
    - credit_history_years: Credit line age in years
    - high_debt: Binary flag for high-risk debt indicators
    - risk_score: Composite risk indicator

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing loan data with required columns:
        - revol.util
        - inq.last.6mths
        - installment
        - log.annual.inc
        - days.with.cr.line
        - dti

    Returns:
    --------
    pd.DataFrame
        DataFrame with 6 new engineered features added

    Examples:
    ---------
    >>> df_enhanced = create_loan_features(df)
    >>> print(df_enhanced.columns)
    """
    tu.print_heading("FEATURE ENGINEERING")

    df_fe = df.copy()

    print("\nCreating new features...")

    # 1. Credit Utilization Ratio (normalized)
    df_fe['credit_util_ratio'] = df_fe['revol.util'] / 100
    print("  + credit_util_ratio: Normalized revolving utilization")

    # 2. Annual Inquiry Rate (project 6 months to annual)
    df_fe['annual_inquiry_rate'] = df_fe['inq.last.6mths'] * 2
    print("  + annual_inquiry_rate: Projected annual inquiries")

    # 3. Debt Burden (monthly payment / annual income)
    df_fe['debt_burden'] = df_fe['installment'] / np.exp(df_fe['log.annual.inc'])
    print("  + debt_burden: Monthly payment / annual income")

    # 4. Credit History in Years
    df_fe['credit_history_years'] = df_fe['days.with.cr.line'] / 365.25
    print("  + credit_history_years: Credit line age in years")

    # 5. High Debt Indicator (binary flag)
    df_fe['high_debt'] = ((df_fe['dti'] > 20) |
                          (df_fe['revol.util'] > 80)).astype(int)
    print("  + high_debt: Binary flag for high debt indicators")

    # 6. Risk Score (composite indicator)
    df_fe['risk_score'] = (
        df_fe['credit_util_ratio'] * 0.3 +
        (df_fe['inq.last.6mths'] / 10) * 0.2 +
        (df_fe['dti'] / 30) * 0.3 +
        (1 - np.clip(df_fe['credit_history_years'] / 20, 0, 1)) * 0.2
    )
    print("  + risk_score: Composite risk indicator")

    original_count = len(df.columns)
    new_count = len(df_fe.columns)
    added_count = new_count - original_count

    print(f"\nOriginal features: {original_count}")
    print(f"Total features after engineering: {new_count}")
    print(f"New features added: {added_count}")

    return df_fe


def get_engineered_feature_names():
    """
    Returns list of feature names created by create_loan_features.

    Returns:
    --------
    list
        List of engineered feature names
    """
    return [
        'credit_util_ratio',
        'annual_inquiry_rate',
        'debt_burden',
        'credit_history_years',
        'high_debt',
        'risk_score'
    ]


def validate_features_for_engineering(df):
    """
    Validate that DataFrame has required columns for feature engineering.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate

    Returns:
    --------
    tuple
        (bool, list) - (is_valid, missing_columns)

    Examples:
    ---------
    >>> valid, missing = validate_features_for_engineering(df)
    >>> if not valid:
    ...     print(f"Missing columns: {missing}")
    """
    required_cols = [
        'revol.util',
        'inq.last.6mths',
        'installment',
        'log.annual.inc',
        'days.with.cr.line',
        'dti'
    ]

    missing = [col for col in required_cols if col not in df.columns]

    return len(missing) == 0, missing


def prepare_data_for_training(df, target_col, categorical_cols=None,
                              test_size=0.3, val_size=0.2,
                              scale_features=True, random_state=42,
                              stratify=True, verbose=True) -> PreparedData:
    """
    Prepare data for machine learning by encoding, splitting, and scaling.

    This function performs the complete preprocessing pipeline:
    1. One-hot encode categorical variables
    2. Separate features and target
    3. Split into train/validation/test sets
    4. Scale features (optional)

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    categorical_cols : list, optional
        List of categorical column names to encode. If None, auto-detects object/category dtypes.
    test_size : float, default=0.3
        Proportion of data for test set (0.0 to 1.0)
    val_size : float, default=0.2
        Proportion of training data for validation set (0.0 to 1.0)
    scale_features : bool, default=True
        Whether to apply StandardScaler to features
    random_state : int, default=42
        Random seed for reproducibility
    stratify : bool, default=True
        Whether to maintain class balance in splits (for classification)
    verbose : bool, default=True
        Whether to print progress messages

    Returns:
    --------
    PreparedData
        A dataclass containing:
        - X_train: Training features (scaled if scale_features=True)
        - X_val: Validation features (scaled if scale_features=True)
        - X_test: Test features (scaled if scale_features=True)
        - y_train: Training target
        - y_val: Validation target
        - y_test: Test target
        - scaler: StandardScaler object (if scale_features=True, else None)
        - encoded_df: DataFrame after encoding (before split)
        - feature_names: List of feature column names

    Examples:
    ---------
    >>> # Basic usage
    >>> data = prepare_data_for_training(df, target_col='not.fully.paid')
    >>> X_train, y_train = data.X_train, data.y_train

    >>> # Custom splits without scaling
    >>> data = prepare_data_for_training(
    ...     df,
    ...     target_col='target',
    ...     categorical_cols=['category1', 'category2'],
    ...     test_size=0.2,
    ...     val_size=0.15,
    ...     scale_features=False
    ... )
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if verbose:
        tu.print_heading("DATA PREPROCESSING")

    df_work = df.copy()

    # 1. Encode categorical variables
    if categorical_cols is None:
        categorical_cols = df_work.select_dtypes(include=['object', 'category']).columns.tolist()
        # Remove target if it's in categorical columns
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

    if verbose and categorical_cols:
        print(f"\n1. Encoding categorical variable(s): {categorical_cols}...")

    if categorical_cols:
        df_encoded = pd.get_dummies(df_work, columns=categorical_cols, drop_first=True, dtype=int)
        if verbose:
            print(f"   ✓ Original features: {df_work.shape[1]}")
            print(f"   ✓ After encoding: {df_encoded.shape[1]}")
    else:
        df_encoded = df_work
        if verbose:
            print("\n1. No categorical variables to encode")

    # 2. Separate features and target
    if verbose:
        print("\n2. Separating features and target...")

    if target_col not in df_encoded.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    if verbose:
        print(f"   ✓ Features shape: {X.shape}")
        print(f"   ✓ Target shape: {y.shape}")

    # 3. Train-test split
    if verbose:
        print(f"\n3. Splitting into train ({(1-test_size)*100:.0f}%) and test ({test_size*100:.0f}%) sets...")

    stratify_y = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    # 4. Train-validation split
    if val_size > 0:
        if verbose:
            print(f"4. Further split training data into train ({(1-val_size)*100:.0f}%) and validation ({val_size*100:.0f}%) sets...")

        stratify_y_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=stratify_y_temp
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = None, None

    if verbose:
        print(f"   ✓ Training set: {X_train.shape[0]:,} samples")
        if X_val is not None:
            print(f"   ✓ Validation set: {X_val.shape[0]:,} samples")
        print(f"   ✓ Test set: {X_test.shape[0]:,} samples")

        # Show class distribution
        print("\n   Class distribution in training set:")
        train_dist = y_train.value_counts(normalize=True) * 100
        for class_label in sorted(train_dist.index):
            print(f"     Class {class_label}: {train_dist[class_label]:.2f}%")

    # 5. Feature scaling
    scaler = None
    if scale_features:
        if verbose:
            print(f"\n5. Scaling features (StandardScaler)...")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if X_val is not None:
            X_val = scaler.transform(X_val)

        if verbose:
            print(f"   ✓ Features scaled to mean=0, std=1")

    # Return results as dataclass
    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        encoded_df=df_encoded,
        feature_names=X.columns.tolist()
    )


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nThis module provides feature engineering functions for loan data.")
    print("\nAvailable functions:")
    print("  - create_loan_features(df)")
    print("  - get_engineered_feature_names()")
    print("  - validate_features_for_engineering(df)")
    print("  - prepare_data_for_training(df, target_col, ...)")
