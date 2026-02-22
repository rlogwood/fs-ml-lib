"""
Model Evaluation Module for Machine Learning Models

This module provides comprehensive functions for evaluating, visualizing,
and optimizing classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Optional, Any
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
    from .utility import get_predictions
    from .model_optimizer import OptimizationMetric, calculate_optimization_metric
except ImportError:
    import text_util as tu
    from utility import get_predictions
    from model_optimizer import OptimizationMetric, calculate_optimization_metric


@dataclass
class ModelEvaluationResult:
    """
    Comprehensive model evaluation results.

    Attributes:
    -----------
    y_pred : np.ndarray
        Binary class predictions
    y_pred_proba : np.ndarray
        Prediction probabilities for positive class
    confusion_matrix : np.ndarray
        Confusion matrix
    cm_fig : matplotlib.figure.Figure
        Confusion matrix figure
    auc : float
        Area under ROC curve
    roc_fig : matplotlib.figure.Figure
        ROC curve figure
    pr_fig : matplotlib.figure.Figure
        Precision-recall curve figure
    best_threshold : float
        Optimal classification threshold
    threshold_df : pd.DataFrame
        DataFrame with metrics at different thresholds
    """
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    confusion_matrix: np.ndarray
    cm_fig: Any  # matplotlib.figure.Figure
    auc: float
    roc_fig: Any  # matplotlib.figure.Figure
    pr_fig: Any  # matplotlib.figure.Figure
    best_threshold: float
    threshold_df: pd.DataFrame


def plot_class_distribution_analysis(df, target_col, labels=None):
    # 1. CLASS IMBALANCE ANALYSIS (CRITICAL)
    tu.print_heading("CLASS DISTRIBUTION ANALYSIS")

    labels = labels or ['Paid (0)', 'Default (1)']
    class_counts = df[target_col].value_counts()
    class_percentages = df[target_col].value_counts(normalize=True) * 100

    print("\nAbsolute Counts:")
    print(f"  Paid (0):     {class_counts[0]:,} loans ({class_percentages[0]:.2f}%)")
    print(f"  Default (1):  {class_counts[1]:,} loans ({class_percentages[1]:.2f}%)")
    paid_debt_ratio=class_counts[0]/class_counts[1]
    imbalanced=paid_debt_ratio>1
    print(tu.bold_and_colored_text(f"Ratio: {paid_debt_ratio:.2f}","red"))
    print("   This significant imbalance requires special handling!")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    #axes[0].bar(['Paid (0)', 'Default (1)'], class_counts.values,
    axes[0].bar(labels, class_counts.values,
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Loans', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution (Absolute)', fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 100, f'{v:,}\n({class_percentages.values[i]:.1f}%)',
                    ha='center', fontweight='bold')

    # Pie chart
    #axes[1].pie(class_counts.values, labels=['Paid (0)', 'Default (1)'],
    axes[1].pie(class_counts.values, labels=labels,
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
               startangle=90, explode=(0, 0.1))
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
    return fig


def plot_training_history(history, metrics=['loss', 'accuracy', 'precision', 'recall', 'auc']):
    """
    Visualize training history with multiple metrics.

    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object from model.fit()
    metrics : list
        List of metrics to plot

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object

    Examples:
    ---------
    >>> fig = plot_training_history(history)
    >>> plt.show()
    """
    tu.print_heading("TRAINING HISTORY VISUALIZATION")

    #fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2  # ceil division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        if metric in history.history:
            axes[idx].plot(history.history[metric], label=f'Training {metric}', linewidth=2)

            val_metric = f'val_{metric}'
            if val_metric in history.history:
                axes[idx].plot(history.history[val_metric],
                             label=f'Validation {metric}',
                             linewidth=2,
                             linestyle='--')

            axes[idx].set_title(f'{metric.capitalize()} over Epochs', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])  # actually remove, not just hide

    #plt.tight_layout()
    #for idx in range(len(metrics), len(axes)):
    #    axes[idx].set_visible(False)

    print("+ Training history visualized")
    #plt.close(fig)
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=['Paid (0)', 'Default (1)'], figsize=(10, 8)):
    """
    Plot confusion matrix with detailed breakdown.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list
        Class labels for display
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object

    Examples:
    ---------
    >>> fig = plot_confusion_matrix(y_test, y_pred)
    >>> plt.show()
    """
    tu.print_heading("MODEL EVALUATION - CONFUSION MATRIX")

    cm = confusion_matrix(y_true, y_pred)

    print(tu.bold_and_colored_text("Confusion Matrix:", tu.Color.ORANGE))
    print(tu.bold_text(str(cm)))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels, #cbar=False,
               cbar_kws={'label': 'Count'},
               annot_kws={'size': 16, 'weight': 'bold'})

    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=14, fontweight='bold')

    # Normalized heatmap
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=labels, yticklabels=labels, #cbar=False)
               cbar_kws={'label': 'Percentage'},
               annot_kws={'size': 16, 'weight': 'bold'})

    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=14, fontweight='bold')

    plt.tight_layout()


    print(tu.bold_and_colored_text("Breakdown:", tu.Color.ORANGE))
    tn, fp, fn, tp = cm.ravel()
    cm_norm *= 100
    print(tu.bold_and_colored_text(
        f"  True Negatives (TN):  {tn:>5,} ({cm_norm[0, 0]:.1f}%) - Correctly predicted as {labels[0]}", tu.Color.GREEN))
    print(tu.bold_and_colored_text(
        f"  False Positives (FP): {fp:>5,} ({cm_norm[0, 1]:.1f}%) - Incorrectly predicted as {labels[1]}", tu.Color.RED))
    print(tu.bold_and_colored_text(
        f"  False Negatives (FN): {fn:>5,} ({cm_norm[1, 0]:.1f}%) - Incorrectly predicted as {labels[0]} (COSTLY!)",
        tu.Color.RED))
    print(tu.bold_and_colored_text(
        f"  True Positives (TP):  {tp:>5,} ({cm_norm[1, 1]:.1f}%) - Correctly predicted as {labels[1]}", tu.Color.GREEN))

    return cm, fig

def print_classification_metrics(y_true, y_pred, y_pred_proba=None, labels=['Paid (0)', 'Default (1)']):
    """
    Print comprehensive classification metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
    labels : list
        Class label names

    Examples:
    ---------
    >>> print_classification_metrics(y_test, y_pred, y_pred_proba)
    """
    tu.print_heading("CLASSIFICATION REPORT")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)

    # Key metrics
    tu.print_heading("KEY PERFORMANCE METRICS")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"  Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (Default):  {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall (Default):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score (Default):   {f1:.4f}")

    if y_pred_proba is not None:
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_true, y_pred_proba)
        print(f"  AUC-ROC:              {auc_score:.4f}")

def plot_roc_curve(y_true, y_pred_proba, figsize=(10, 8), usr_title=None):
    """
    Plot ROC curve with AUC score.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    float
        AUC-ROC score

    Examples:
    ---------
    >>> fig, auc_score = plot_roc_curve(y_test, y_pred_proba)
    >>> plt.show()
    """
    tu.print_heading("ROC CURVE ANALYSIS")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                fontsize=14, fontweight='bold')
    title = "ROC Curve"
    if usr_title:
        title += f" - {usr_title}"

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print(tu.bold_text(f"\nAUC-ROC Score: {roc_auc:.4f}"))
    if roc_auc >= 0.9:
        interpretation = "Excellent discrimination ability"
    elif roc_auc >= 0.8:
        interpretation = "Good discrimination ability"
    elif roc_auc >= 0.7:
        interpretation = "Fair discrimination ability"
    else:
        interpretation = "Poor discrimination ability"
    print(tu.bold_text(f"  Interpretation: {interpretation}"))

    return fig, roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba, figsize=(10, 8), usr_title=None):
    """
    Plot Precision-Recall curve.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object

    Examples:
    ---------
    >>> fig = plot_precision_recall_curve(y_true, y_pred_proba)
    >>> plt.show()
    """
    tu.print_heading("PRECISION-RECALL CURVE")

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')

    plt.axhline(y=y_true.sum() / len(y_true), color='red', linestyle='--', linewidth=2,
                label=f'Baseline (No Skill) = {y_true.sum() / len(y_true):.4f}')

    ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    title = "Precision-Recall Curve"
    if usr_title is not None:
        title += f" - {usr_title}"

    ax.set_title(title, fontsize=16, fontweight='bold')
    #ax.set_xlim([0.0, 1.0])
    #ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()
    print("\n+ Precision-Recall curve plotted")

    return fig

def optimize_threshold(y_true, y_pred_proba, thresholds_to_test=[0.3, 0.4, 0.5, 0.6, 0.7],
                       metric=OptimizationMetric.F1, roc_auc=None, verbose=True):
    """
    Find optimal classification threshold by testing multiple values.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    thresholds_to_test : list
        List of threshold values to test
    metric : OptimizationMetric
        Metric to optimize (e.g., OptimizationMetric.F1, OptimizationMetric.RECALL_WEIGHTED)
    roc_auc : float, optional
        ROC AUC score (required for RECALL_WEIGHTED metric, as it doesn't change with threshold)
    verbose : bool
        Print results table

    Returns:
    --------
    dict
        Best threshold and corresponding metrics
    pd.DataFrame
        DataFrame with all results

    Examples:
    ---------
    >>> best_result, results_df = optimize_threshold(y_test, y_pred_proba,
    ...                                               metric=OptimizationMetric.F1)
    >>> print(f"Best threshold: {best_result['threshold']}")

    >>> # Using recall_weighted
    >>> best_result, results_df = optimize_threshold(y_test, y_pred_proba,
    ...                                               metric=OptimizationMetric.RECALL_WEIGHTED,
    ...                                               roc_auc=0.75)
    """
    if not isinstance(metric, OptimizationMetric):
        raise TypeError(f"metric must be OptimizationMetric enum, got {type(metric).__name__}")

    metric_name = metric.value

    if verbose:
        tu.print_heading("THRESHOLD OPTIMIZATION")
        print(tu.italic_text("Testing different classification thresholds:"))
        print("=" * 60)
        print(tu.bold_text(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}"))

    results = []

    for threshold in thresholds_to_test:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred_thresh)
        prec = precision_score(y_true, y_pred_thresh, zero_division=0)
        rec = recall_score(y_true, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)

        # Calculate the optimization metric for this threshold
        opt_metric_value = calculate_optimization_metric(
            metric,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc
        )

        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'optimization_metric': opt_metric_value
        })

        if verbose:
            print(f"{threshold:<12.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

    results_df = pd.DataFrame(results)

    # Find best threshold based on optimization metric
    best_idx = results_df['optimization_metric'].idxmax()
    best_result = results_df.iloc[best_idx].to_dict()

    if verbose:
        print("-" * 60)
        print(tu.bold_text(f"Best Threshold (by {metric_name.upper()}): {best_result['threshold']}"))
        print(f"  F1-Score: {best_result['f1']:.4f}")
        print(f"  Precision: {best_result['precision']:.4f}")
        print(f"  Recall: {best_result['recall']:.4f}")
        if metric == OptimizationMetric.RECALL_WEIGHTED:
            print(f"  Recall-Weighted Score: {best_result['optimization_metric']:.4f}")
        print("=" * 60 + "\n")

    return best_result, results_df

def evaluate_model_comprehensive(model, X_test, y_test, class_names=['Paid', 'Default'],
                                 usr_title=None, optimize_for=OptimizationMetric.F1):
    """
    Perform comprehensive model evaluation with all visualizations.

    Parameters:
    -----------
    model : keras.Model or sklearn model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    class_names : list
        Class label names
    usr_title : str, optional
        Custom title for plots
    optimize_for : OptimizationMetric
        Metric to optimize threshold for (default: OptimizationMetric.F1)
        Options: OptimizationMetric.ACCURACY, PRECISION, RECALL, F1, ROC_AUC, RECALL_WEIGHTED

    Returns:
    --------
    ModelEvaluationResult
        Dataclass containing all metrics and figures

    Examples:
    ---------
    >>> results = evaluate_model_comprehensive(model, X_test, y_test)
    >>> print(f"AUC: {results.auc:.4f}")
    >>> print(f"Best threshold: {results.best_threshold:.3f}")

    >>> # Using recall_weighted optimization
    >>> results = evaluate_model_comprehensive(model, X_test, y_test,
    ...                                        optimize_for=OptimizationMetric.RECALL_WEIGHTED)
    """
    if not isinstance(optimize_for, OptimizationMetric):
        raise TypeError(f"optimize_for must be OptimizationMetric enum, got {type(optimize_for).__name__}")

    # Get predictions using shared utility
    y_pred, y_pred_proba = get_predictions(model, X_test, threshold=0.5, verbose=0)

    # 1. Confusion Matrix
    conf_matrix, cm_fig = plot_confusion_matrix(y_test, y_pred, labels=class_names)

    # 2. Classification Metrics
    print_classification_metrics(y_test, y_pred, y_pred_proba,
                                 labels=[f'{class_names[0]} (0)', f'{class_names[1]} (1)'])

    # 3. ROC Curve
    roc_fig, auc_score = plot_roc_curve(y_test, y_pred_proba)

    # 4. Precision-Recall Curve
    pr_fig = plot_precision_recall_curve(y_test, y_pred_proba, usr_title=usr_title)

    # 5. Threshold Optimization
    best_threshold_dict, threshold_df = optimize_threshold(
        y_test, y_pred_proba, metric=optimize_for, roc_auc=auc_score
    )

    # Create and return dataclass instance
    return ModelEvaluationResult(
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        confusion_matrix=conf_matrix,
        cm_fig=cm_fig,
        auc=auc_score,
        roc_fig=roc_fig,
        pr_fig=pr_fig,
        best_threshold=best_threshold_dict['threshold'],
        threshold_df=threshold_df
    )

if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("=" * 50)
    print("\nThis module provides comprehensive model evaluation functions.")
    print("\nAvailable functions:")
    print("  - plot_training_history(history)")
    print("  - plot_confusion_matrix(y_true, y_pred)")
    print("  - print_classification_metrics(y_true, y_pred, y_pred_proba)")
    print("  - plot_roc_curve(y_true, y_pred_proba)")
    print("  - plot_precision_recall_curve(y_true, y_pred_proba)")
    print("  - optimize_threshold(y_true, y_pred_proba)")
    print("  - evaluate_model_comprehensive(model, X_test, y_test)")
