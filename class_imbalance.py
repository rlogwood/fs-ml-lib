from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


# handle lib imports
# try:
#     # When imported as part of a package
#     from . import text_util as tu, utility as utl, imputer as im
# except ImportError:
#     # When run as a standalone script
#     import text_util as tu
#     import utility as utl
#     import imputer as im

import lib.text_util as tu
#import utility as utl
#import imputer as im

@dataclass
class ClassAnalysisItem:
    class_label: Any
    count: int
    percentage: float
    ratio_to_majority: float
    ratio_to_minority: float


@dataclass
class ImbalanceAnalysisResult:
    n_classes: int
    total_samples: int
    majority_class: Any
    majority_count: int
    minority_class: Any
    minority_count: int
    minority_percentage: str
    imbalance_ratio: str
    severity: str
    recommended_action: str
    class_analysis: list[ClassAnalysisItem]

    def display(self):
        tu.print_heading("CLASS IMBALANCE ANALYSIS")
        tu.print_sub_heading(f"Number of classes: {self.n_classes}")
        tu.print_sub_heading(f"Total samples: {self.total_samples:,}")
        print(f"\nClass Distribution:")
        print("-" * 70)

        sorted_analysis = sorted(self.class_analysis, key=lambda x: x.count, reverse=True)
        for item in sorted_analysis:
            bar_length = int(item.percentage / 2)
            bar = "█" * bar_length
            print(f"  Class {item.class_label:>10}: {item.count:>8,} ({item.percentage:>6.2f}%) {bar}")

        print("\n" + "-" * 70)
        print(f"Majority class: {self.majority_class} ({self.majority_count:,} samples)")
        print(f"Minority class: {self.minority_class} ({self.minority_count:,} samples)")
        print(f"Imbalance ratio: {self.imbalance_ratio}")
        print(f"\nSeverity: {self.severity}")
        print(f"Recommended action: {self.recommended_action}")


def check_imbalance(class_counts, verbose=True):
    """
    Check if dataset is imbalanced for binary or multi-class problems

    Parameters:
    -----------
    class_counts : dict, pd.Series, or array-like
        Counts of each class {class_label: count} or array of counts
    verbose : bool
        Whether to print detailed information

    Returns:
    --------
    dict with comprehensive imbalance analysis
    """
    # Convert to dict format
    if isinstance(class_counts, pd.Series):
        class_dict = class_counts.to_dict()
        counts = class_counts.values
    elif isinstance(class_counts, dict):
        class_dict = class_counts
        counts = np.array(list(class_counts.values()))
    else:
        counts = np.array(class_counts)
        class_dict = {i: count for i, count in enumerate(counts)}

    # Basic statistics
    total = counts.sum()
    n_classes = len(counts)
    majority_count = counts.max()
    minority_count = counts.min()
    ratio = majority_count / minority_count

    # Calculate imbalance ratio for each class vs majority
    majority_idx = counts.argmax()
    minority_idx = counts.argmin()

    # Determine severity based on worst-case ratio
    if ratio < 1.5:
        severity = "Balanced ✅"
        action = "No special handling needed"
    elif ratio < 3:
        severity = "Slight Imbalance ⚠️"
        action = "Monitor metrics closely"
    elif ratio < 9:
        severity = "Moderate Imbalance 🟡"
        action = "Use class weights or resampling"
    elif ratio < 99:
        severity = "Severe Imbalance 🟠"
        action = "Special techniques required (SMOTE, heavy class weights)"
    else:
        severity = "Extreme Imbalance 🔴"
        action = "Consider anomaly detection approaches"

    # Per-class analysis
    class_analysis = []
    class_labels = list(class_dict.keys())

    for i, (label, count) in enumerate(class_dict.items()):
        percentage = (count / total) * 100
        ratio_to_majority = majority_count / count if count > 0 else np.inf
        ratio_to_minority = count / minority_count if minority_count > 0 else np.inf

        class_analysis.append(ClassAnalysisItem(
            class_label=label,
            count=count,
            percentage=percentage,
            ratio_to_majority=ratio_to_majority,
            ratio_to_minority=ratio_to_minority
        ))

    result = ImbalanceAnalysisResult(
        n_classes=n_classes,
        total_samples=int(total),
        majority_class=class_labels[majority_idx],
        majority_count=int(majority_count),
        minority_class=class_labels[minority_idx],
        minority_count=int(minority_count),
        minority_percentage=f"{(minority_count / total) * 100:.2f}%",
        imbalance_ratio=f"{ratio:.2f}:1",
        severity=severity,
        recommended_action=action,
        class_analysis=class_analysis
    )

    if verbose:
        result.display()

    return result
