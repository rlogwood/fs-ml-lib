from dataclasses import dataclass
from typing import Any
from typing import Union, Sequence
import numpy as np
import pandas as pd
from pprint import pprint
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#import text_util as tu
# class_imbalance.py
try:
    import lib.text_util as tu
except ImportError:
    import text_util as tu


@dataclass
class CorrelationAnalysisConfig:
    correlation_threshold: float = 0.7


@dataclass
class CorrelationAnalysisResult:
    features_to_drop: list[str]
    high_corr_pairs: list[tuple[str, str, float]]
    features_to_drop: list[str]


def find_high_corr_pairs(df: pd.DataFrame, correlation_threshold: float = 0.7) -> list[tuple[str, str, float]]:
    corr_matrix = df.corr().abs()
    #print(corr_matrix)
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    #print(upper_triangle)
    upper_corr = corr_matrix.where(upper_triangle)
    #print(upper_corr)

    high_corr_pairs = []
    for col in upper_corr.columns:
        for idx in upper_corr.index:
            corr_value = upper_corr.loc[idx, col]
            if pd.notna(corr_value) and corr_value > correlation_threshold:
                high_corr_pairs.append((idx, col, corr_value))

    #print(f"Highly Correlated Feature Pairs Found: {high_corr_pairs}")
    return high_corr_pairs


def drop_highly_correlated_features(df: pd.DataFrame, target_col: str, correlation_threshold: float = 0.7):
    print(tu.bold_text("\nDROPPING HIGHLY CORRELATED FEATURES"))


    # Get correlation matrix for numeric features (excluding target)
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop(target_col)

    print(f"\n- Number of numeric features: {len(numeric_features)}")
    print("\t",end="")
    print("\n\t".join(map(str, numeric_features.tolist())))

    print(f"\nDropping highly correlated features with threshold {correlation_threshold} for target column '{target_col}'")
    #corr_matrix = df[numeric_features].corr().abs()
    high_corr_pairs = find_high_corr_pairs(df[numeric_features], correlation_threshold)

    print(f"Number of highly correlated pairs found: {len(high_corr_pairs)}\n\n")


    if not high_corr_pairs:
        #print("  No highly correlated pairs found above threshold.")
        features_to_drop = []
    else:
        # Determine which features to drop
        # Strategy: For each pair, drop the feature with lower correlation to target
        features_to_drop = []
        target_corr = df[numeric_features].corrwith(df[target_col]).abs()

        for feat1, feat2, corr_val in high_corr_pairs:
            corr1 = target_corr[feat1]
            corr2 = target_corr[feat2]

            print(f"highly correlated: {feat1} <-> {feat2}: {corr_val:.4f}")
            #print(f"  {feat1} corr to target: {corr1:.4f}")
            #print(f"  {feat2} corr to target: {corr2:.4f}")
            # Drop the feature with lower correlation to target
            if corr1 < corr2:
                drop_feat = feat1
                keep_feat = feat2
            else:
                drop_feat = feat2
                keep_feat = feat1

            if drop_feat not in features_to_drop:
                features_to_drop.append(drop_feat)
                print(f"  Dropping '{drop_feat}' (target corr: {target_corr[drop_feat]:.4f})")
                print(f"  Keeping '{keep_feat}' (target corr: {target_corr[keep_feat]:.4f})")
                print()

    # Drop the identified features
    print(tu.bold_text("APPLYING FEATURE SELECTION"))

    print(f"\nOriginal number of features: {df.shape[1]}")
    print(f"Features to drop: {features_to_drop if features_to_drop else 'None'}")

    if features_to_drop:
        df = df.drop(columns=features_to_drop)
        print(f"Features after dropping: {df.shape[1]}")
        print(f"\n✓ Dropped {len(features_to_drop)} highly correlated feature(s): {features_to_drop}")
    else:
        pass
        print("\n✓ No features needed to be dropped (no high correlations found)")

    #print(df)
    #print(f"Dropping features: {features_to_drop}")
    return df, features_to_drop
    #print("\nRemaining features:")
    #print(df.columns.tolist())


def analyze_correlation(df, target_col=None, figsize=(14, 10),
                        cmap='coolwarm', fmt='.2f',
                        mask_upper=True, print_target_corr=True):
    """
    Perform correlation analysis on numeric features and visualize with heatmap.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str, optional
        Target column name to show correlations with. If None, no target correlation printed.
    figsize : tuple, default=(14, 10)
        Figure size for the heatmap
    cmap : str, default='coolwarm'
        Colormap for the heatmap
    fmt : str, default='.2f'
        String formatting for annotation values
    mask_upper : bool, default=True
        Whether to mask the upper triangle of the correlation matrix
    print_target_corr : bool, default=True
        Whether to print correlation with target variable

    Returns:
    --------
    correlation_matrix : pd.DataFrame
        Correlation matrix of numeric features
    target_corr : pd.Series or None
        Correlation with target variable (if target_col provided)
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()

    # Print correlation with target if specified
    target_corr = None
    if target_col and target_col in correlation_matrix.columns and print_target_corr:
        target_corr = correlation_matrix[target_col].sort_values(ascending=False)
        print(f"\nCorrelation with Target ({target_col}):")
        print(target_corr)

    # Create visualization
    plt.figure(figsize=figsize)

    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create heatmap
    ax = sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )

    if mask_upper:
        ax.set_facecolor("white")  # hide the upper half of the heatmap

    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    return correlation_matrix, target_corr
