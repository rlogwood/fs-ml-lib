"""
Visualization Helpers

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

try:
    from . import text_util as tu
except ImportError:
    import text_util as tu

def plot_key_features_distribution(df, target_col, class_labels, class_names, colors=None, features_to_plot=None):
    """
    feature : str - feature column name
    target_col : str - target variable column name
    class_labels : list - class label values (e.g., [0, 1])
    class_names : list - display names for classes (e.g., ['Paid', 'Default'])
    colors : list - colors for each class
    """

    # 3. KEY FEATURE DISTRIBUTIONS
    tu.print_heading("KEY FEATURE DISTRIBUTIONS")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Distribution of Key Features by Default Status',
                 fontsize=16, fontweight='bold', y=1.00)

    if features_to_plot is None:
        features_to_plot = ['fico', 'int.rate', 'dti', 'revol.util', 'inq.last.6mths', 'delinq.2yrs']


    if colors is None:
        colors = ['#2ecc71', '#e74c3c']

    for idx, feature in enumerate(features_to_plot):
        row = idx // 3
        col = idx % 3

        # Plot distributions for both classes
        df[df[target_col] == class_labels[0]][feature].hist(ax=axes[row, col], bins=30,
                                                    alpha=0.6, label=class_names[0],
                                                    color=colors[0], edgecolor='black')
        df[df[target_col] == class_labels[1]][feature].hist(ax=axes[row, col], bins=30,
                                                    alpha=0.6, label=class_names[1],
                                                    color=colors[1], edgecolor='black')

        axes[row, col].set_xlabel(feature, fontsize=11, fontweight='bold')
        axes[row, col].set_ylabel('Frequency', fontsize=11)
        axes[row, col].legend()
        axes[row, col].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✓ Feature distributions plotted")