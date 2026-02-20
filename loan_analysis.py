"""
Lending Club Loan Analysis helpers

This module provides functions lending club loan data analysis.
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

def plot_loan_purpose_analysis(df):
    # 2. LOAN PURPOSE DISTRIBUTION
    tu.print_heading("LOAN PURPOSE ANALYSIS")

    purpose_counts = df['purpose'].value_counts()
    print("\nLoan Purposes:")
    for purpose, count in purpose_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {purpose:.<25} {count:>5,} ({pct:>5.1f}%)")

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='purpose', order=purpose_counts.index,
                  hue='not.fully.paid', palette={0: '#2ecc71', 1: '#e74c3c'})
    plt.title('Loan Purpose Distribution by Default Status', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Loans', fontsize=12)
    plt.ylabel('Purpose', fontsize=12)
    plt.legend(title='Status', labels=['Paid', 'Default'])
    plt.tight_layout()
    plt.show()
