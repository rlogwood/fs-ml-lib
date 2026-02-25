"""
model_optimizer.py - Automated model optimization and comparison
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

try:
    import lib.text_util as tu
    from lib.utility import get_predictions
except ImportError:
    from utility import get_predictions
    import text_util as tu


class ImbalanceStrategy(Enum):
    """
    Enumeration of available imbalance handling strategies.

    Strategies:
    -----------
    NONE : No imbalance handling
    SMOTE_FULL : Full SMOTE oversampling (1:1 balance)
    SMOTE_PARTIAL : Partial SMOTE oversampling (custom ratio)
    CLASS_WEIGHTS : Use class weights only
    SMOTE_PARTIAL_WEIGHTS : Combined SMOTE + class weights
    """
    NONE = "none"
    SMOTE_FULL = "smote_full"
    SMOTE_PARTIAL = "smote_partial"
    CLASS_WEIGHTS = "class_weights"
    SMOTE_PARTIAL_WEIGHTS = "smote_partial+weights"

    def __str__(self):
        return self.value


class OptimizationMetric(Enum):
    """
    Enumeration of available optimization metrics.

    Metrics:
    --------
    ACCURACY : Classification accuracy
    PRECISION : Precision score
    RECALL : Recall score
    F1 : F1 score (harmonic mean of precision and recall)
    ROC_AUC : Area under the ROC curve
    RECALL_WEIGHTED : Weighted combination (0.3*roc_auc + 0.6*recall + 0.1*f1)
    """
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    RECALL_WEIGHTED = "recall_weighted"

    def __str__(self):
        return self.value


@dataclass
class ValidationMetrics:
    """
    Validation metrics for model evaluation.

    Attributes:
    -----------
    accuracy : float
        Classification accuracy
    precision : float
        Precision score
    recall : float
        Recall score
    f1 : float
        F1 score (harmonic mean of precision and recall)
    roc_auc : float
        Area under the ROC curve
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for backwards compatibility"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc
        }

    def __getitem__(self, key: str) -> float:
        """Allow dict-style access for backwards compatibility"""
        return getattr(self, key)


@dataclass
class ImbalanceTrainingResult:
    """Results from training with imbalance handling"""
    strategy: Union[ImbalanceStrategy, str]
    history: Any
    model: Any
    X_val: np.ndarray
    y_val: np.ndarray
    X_train_final: np.ndarray
    y_train_final: np.ndarray
    X_train_original: np.ndarray
    y_train_original: np.ndarray
    class_weight_dict: Optional[Dict[int, float]]
    smote_ratio: Optional[float]
    samples_before: int
    samples_after: int
    class_dist_before: Dict[int, int]
    class_dist_after: Dict[int, int]

    # Validation metrics
    val_metrics: ValidationMetrics = None

    def label_for_class_value(self, class_value: int, class_labels: dict = None) -> str:
        if class_labels and class_value in class_labels:
            label = f"Class {class_value} ({class_labels[class_value]})"
        else:
            label = f"Class {class_value}"
        #print("label for class value:{class_value} is:{label}")
        return label

    def summary(self, class_labels: dict = None):
        """Print a summary of the training configuration"""
        strategy_name = self.strategy.value if isinstance(self.strategy, ImbalanceStrategy) else self.strategy
        print(f"\n{'=' * 70}")
        print(f"IMBALANCE HANDLING SUMMARY: {strategy_name}")
        print(f"{'=' * 70}")
        print(f"Samples: {self.samples_before:,} → {self.samples_after:,}")
        print(f"\nClass Distribution Before:")
        for cls, count in self.class_dist_before.items():
            pct = count / self.samples_before * 100
            label = self.label_for_class_value(cls, class_labels)
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        print(f"\nClass Distribution After:")
        for cls, count in self.class_dist_after.items():
            pct = count / self.samples_after * 100
            label = self.label_for_class_value(cls, class_labels)
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        if self.class_weight_dict:
            print(f"\nClass Weights Applied:")
            for cls, weight in self.class_weight_dict.items():
                label = self.label_for_class_value(cls, class_labels)
                print(f"  {label}: {weight:.4f}")
        if self.smote_ratio:
            print(f"\nSMOTE Ratio: {self.smote_ratio}")

        if self.val_metrics:
            print(f"\nValidation Metrics:")
            for metric, value in vars(self.val_metrics).items():
                print(f"  {metric}: {value:.4f}")
        print(f"{'=' * 70}\n")

    def display_markdown(self, class_labels: dict = None) -> str:
        """Generate markdown formatted summary of class weights.

        Parameters:
        -----------
        class_labels : dict, optional
            Mapping of class values to human-readable labels {class_value: "Display Name"}
            Example: {0: "Paid", 1: "Default"}

        Returns:
        --------
        str: Markdown formatted string with class weights information
        """
        md = ""
        if self.class_weight_dict:
            md += "**Calculated Class Weights**:\n"
            for class_value, weight in self.class_weight_dict.items():
                label = self.label_for_class_value(class_value, class_labels)
                md += f"- {label}: {weight:.3f}\n"
            md += "\n"
        return md


@dataclass
class OptimizationComparison:
    """Comparison results across all strategies"""
    results: Dict[str, ImbalanceTrainingResult]
    comparison_df: pd.DataFrame
    best_strategy: str
    best_metric: str
    ranking: List[tuple]  # [(strategy, score), ...]

    def print_comparison(self):
        """Print formatted comparison table"""
        print(f"\n{'=' * 70}")
        print("STRATEGY COMPARISON")
        print(f"{'=' * 70}")
        print(self.comparison_df.to_string(index=True))
        print(f"\n{'=' * 70}")
        print(f"🏆 BEST STRATEGY: {self.best_strategy}")
        print(f"   Optimized for: {self.best_metric}")
        print(f"{'=' * 70}\n")

    def get_best_model(self):
        """Return the best trained model"""
        return self.results[self.best_strategy].model


def calculate_recall_weighted(roc_auc, recall, f1):
    """
    Calculate weighted recall score.

    Formula: 0.3*roc_auc + 0.6*recall + 0.1*f1

    This metric emphasizes recall (60%) while considering AUC (30%) and F1 (10%),
    useful for scenarios where catching positive cases is critical.

    Parameters:
    -----------
    roc_auc : float
        ROC AUC score
    recall : float
        Recall score
    f1 : float
        F1 score

    Returns:
    --------
    float
        Weighted recall score
    """
    return 0.3 * roc_auc + 0.6 * recall + 0.1 * f1


def calculate_optimization_metric(metric_type, accuracy=None, precision=None, recall=None, f1=None, roc_auc=None):
    """
    Calculate optimization metric value based on OptimizationMetric type.

    Parameters:
    -----------
    metric_type : OptimizationMetric
        Type of metric to calculate (must be OptimizationMetric enum)
    accuracy : float, optional
        Accuracy score
    precision : float, optional
        Precision score
    recall : float, optional
        Recall score
    f1 : float, optional
        F1 score
    roc_auc : float, optional
        ROC AUC score

    Returns:
    --------
    float
        Calculated metric value

    Raises:
    -------
    TypeError
        If metric_type is not OptimizationMetric enum
    ValueError
        If required metric values are missing
    """
    if not isinstance(metric_type, OptimizationMetric):
        raise TypeError(f"metric_type must be OptimizationMetric enum, got {type(metric_type).__name__}")

    # Calculate based on metric type
    if metric_type == OptimizationMetric.ACCURACY:
        if accuracy is None:
            raise ValueError("accuracy value is required for ACCURACY metric")
        return accuracy
    elif metric_type == OptimizationMetric.PRECISION:
        if precision is None:
            raise ValueError("precision value is required for PRECISION metric")
        return precision
    elif metric_type == OptimizationMetric.RECALL:
        if recall is None:
            raise ValueError("recall value is required for RECALL metric")
        return recall
    elif metric_type == OptimizationMetric.F1:
        if f1 is None:
            raise ValueError("f1 value is required for F1 metric")
        return f1
    elif metric_type == OptimizationMetric.ROC_AUC:
        if roc_auc is None:
            raise ValueError("roc_auc value is required for ROC_AUC metric")
        return roc_auc
    elif metric_type == OptimizationMetric.RECALL_WEIGHTED:
        if roc_auc is None or recall is None or f1 is None:
            raise ValueError("roc_auc, recall, and f1 values are required for RECALL_WEIGHTED metric")
        return calculate_recall_weighted(roc_auc, recall, f1)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def train_with_imbalance_handling(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        strategy: Union[ImbalanceStrategy, str] = ImbalanceStrategy.SMOTE_PARTIAL,
        smote_ratio=0.5,
        class_weight_dict=None,
        auto_calculate_weights=True,
        epochs=50,
        batch_size=256,
        callbacks=None,
        random_state=42,
        train_fn=None,
        verbose=True,
        model_verbosity=1
) -> ImbalanceTrainingResult:
    """
    Train a model with different imbalance handling strategies.

    Parameters:
    -----------
    model : keras.Model
        The neural network model to train
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    strategy : ImbalanceStrategy or str
        Imbalance handling strategy (use ImbalanceStrategy enum or string):
        - ImbalanceStrategy.NONE or 'none': No imbalance handling
        - ImbalanceStrategy.SMOTE_FULL or 'smote_full': Full SMOTE (1:1 balance)
        - ImbalanceStrategy.SMOTE_PARTIAL or 'smote_partial': Partial SMOTE (custom ratio)
        - ImbalanceStrategy.CLASS_WEIGHTS or 'class_weights': Only class weights
        - ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS or 'smote_partial+weights': Combined approach
    smote_ratio : float
        Ratio for partial SMOTE (default 0.5)
    class_weight_dict : dict, optional
        Pre-calculated class weights
    auto_calculate_weights : bool
        Auto-calculate class weights when needed
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training (default 256)
    callbacks : list, optional
        Keras callbacks
    random_state : int
        Random seed
    train_fn : callable, optional
        Custom training function. If None, uses default
    verbose : bool
        Print detailed information

    Returns:
    --------
    ImbalanceTrainingResult
        Dataclass containing all training results and metadata
    """

    # Convert string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = ImbalanceStrategy(strategy)
        except ValueError:
            valid_strategies = [s.value for s in ImbalanceStrategy]
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: {', '.join(valid_strategies)}"
            )

    # Store original data info
    samples_before = len(X_train)
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist_before = dict(zip(unique.astype(int), counts.astype(int)))

    # strategies ImbalanceStrategy.CLASS_WEIGHTS, ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS require class weights
    if strategy in [ImbalanceStrategy.CLASS_WEIGHTS, ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS]:
        # Calculate class weights if not supplied
        if auto_calculate_weights:
            if class_weight_dict is not None:
                raise ValueError(f"class_weight_dict cannot be provided when auto_calculate_weights=True")

            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes.astype(int), weights))
            if verbose:
                print(f"\n📊 Auto-calculated class weights: {class_weight_dict}")
    else:
        # raise error for other strategies, class_weight_dict is not required
        raise ValueError(
            f"Class weights not supported for strategy {strategy}, use ImbalanceStrategy.CLASS_WEIGHTS or ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS"
        )

    # Apply imbalance handling strategy
    if strategy == ImbalanceStrategy.NONE:
        X_train_final, y_train_final = X_train, y_train
        weights = None
        smote_ratio_used = None

    elif strategy == ImbalanceStrategy.SMOTE_FULL:
        smote = SMOTE(sampling_strategy=1.0, random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
        weights = None
        smote_ratio_used = 1.0

    elif strategy == ImbalanceStrategy.SMOTE_PARTIAL:
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
        weights = None
        smote_ratio_used = smote_ratio

    elif strategy == ImbalanceStrategy.CLASS_WEIGHTS:
        X_train_final, y_train_final = X_train, y_train
        weights = class_weight_dict
        smote_ratio_used = None

    elif strategy == ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS:
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
        weights = class_weight_dict
        smote_ratio_used = smote_ratio

    else:
        valid_strategies = [s.value for s in ImbalanceStrategy]
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: {', '.join(valid_strategies)}"
        )

    # Get final class distribution
    samples_after = len(X_train_final)
    unique, counts = np.unique(y_train_final, return_counts=True)
    class_dist_after = dict(zip(unique.astype(int), counts.astype(int)))

    if verbose:
        strategy_name = strategy.value if isinstance(strategy, ImbalanceStrategy) else strategy
        print(f"\n📊 STRATEGY: {strategy_name}")
        print(f"   Samples: {samples_before:,} → {samples_after:,}")
        print(f"   Class dist: {class_dist_before} → {class_dist_after}")

    # Train model using provided function or default
    if train_fn is None:
        # Import here to avoid circular dependency
        try:
            from lib.model_trainer import train_model_with_class_weights
        except ImportError:
            from model_trainer import train_model_with_class_weights
        history = train_model_with_class_weights(
            model, X_train_final, y_train_final, X_val, y_val,
            weights, epochs=epochs, batch_size=batch_size, callbacks=callbacks, model_verbosity=model_verbosity
        )
    else:
        # Custom train function - don't pass model_verbosity (may not be supported)
        history = train_fn(
            model, X_train_final, y_train_final, X_val, y_val,
            weights, epochs, callbacks
        )

    # Calculate validation metrics using shared prediction utility
    y_val_pred, y_val_pred_proba = get_predictions(model, X_val, threshold=0.5, verbose=model_verbosity)

    val_metrics = ValidationMetrics(
        accuracy=accuracy_score(y_val, y_val_pred),
        precision=precision_score(y_val, y_val_pred, zero_division=0),
        recall=recall_score(y_val, y_val_pred, zero_division=0),
        f1=f1_score(y_val, y_val_pred, zero_division=0),
        roc_auc=roc_auc_score(y_val, y_val_pred_proba)
    )

    # Create result object
    result = ImbalanceTrainingResult(
        strategy=strategy,
        history=history,
        model=model,
        X_val=X_val,
        y_val=y_val,
        X_train_final=X_train_final,
        y_train_final=y_train_final,
        X_train_original=X_train,
        y_train_original=y_train,
        class_weight_dict=weights,
        smote_ratio=smote_ratio_used,
        samples_before=samples_before,
        samples_after=samples_after,
        class_dist_before=class_dist_before,
        class_dist_after=class_dist_after,
        val_metrics=val_metrics
    )

    return result


def optimize_imbalance_strategy(
        model_builder: Callable,
        X_train,
        y_train,
        X_val,
        y_val,
        strategies: Optional[List[Union[ImbalanceStrategy, str]]] = None,
        smote_ratios=None,
        optimize_for: Union[OptimizationMetric, str] = OptimizationMetric.F1,
        epochs=50,
        batch_size=256,
        callbacks=None,
        random_state=42,
        train_fn=None,
        verbose=True,
        model_verbosity=1
) -> OptimizationComparison:
    """
    Run all imbalance handling strategies and compare results.

    Parameters:
    -----------
    model_builder : callable
        Function that returns a fresh model instance.
        Signature: model_builder() -> keras.Model
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    strategies : list of ImbalanceStrategy or str, optional
        List of strategies to try (enum or string values). If None, tries all strategies.
    smote_ratios : list, optional
        List of SMOTE ratios to try for partial strategies
    optimize_for : OptimizationMetric or str
        Metric to optimize (use OptimizationMetric enum or string):
        - OptimizationMetric.ACCURACY or 'accuracy': Classification accuracy
        - OptimizationMetric.PRECISION or 'precision': Precision score
        - OptimizationMetric.RECALL or 'recall': Recall score
        - OptimizationMetric.F1 or 'f1': F1 score
        - OptimizationMetric.ROC_AUC or 'roc_auc': Area under ROC curve
        - OptimizationMetric.RECALL_WEIGHTED or 'recall_weighted':
          Weighted score (0.3*roc_auc + 0.6*recall + 0.1*f1)
    epochs : int
        Training epochs per strategy
    batch_size : int
        Batch size for training (default 256)
    callbacks : list, optional
        Keras callbacks
    random_state : int
        Random seed
    train_fn : callable, optional
        Custom training function. If None, uses default Keras training.
        Signature: train_fn(model, X_train, y_train, X_val, y_val, class_weights, epochs, callbacks)
    verbose : bool
        Print progress

    Returns:
    --------
    OptimizationComparison
        Comparison results with best strategy identified
    """

    # Convert string to enum if needed for optimize_for
    if isinstance(optimize_for, str):
        try:
            optimize_for = OptimizationMetric(optimize_for)
        except ValueError:
            valid_metrics = [m.value for m in OptimizationMetric]
            raise ValueError(
                f"Unknown metric: {optimize_for}. "
                f"Choose from: {', '.join(valid_metrics)}"
            )

    # Convert to string for DataFrame operations
    optimize_for_str = optimize_for.value if isinstance(optimize_for, OptimizationMetric) else optimize_for

    if strategies is None:
        strategies = [
            ImbalanceStrategy.NONE,
            ImbalanceStrategy.SMOTE_FULL,
            ImbalanceStrategy.SMOTE_PARTIAL,
            ImbalanceStrategy.CLASS_WEIGHTS,
            ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS
        ]
    else:
        # Convert strings to enums if needed
        converted_strategies = []
        for s in strategies:
            if isinstance(s, str):
                converted_strategies.append(ImbalanceStrategy(s))
            else:
                converted_strategies.append(s)
        strategies = converted_strategies

    if smote_ratios is None:
        smote_ratios = [0.5]

    results = {}

    print(f"\n{'=' * 70}")
    print(f"🔬 OPTIMIZING IMBALANCE HANDLING STRATEGY")
    print(f"   Optimizing for: {optimize_for_str}")
    print(f"   Strategies to test: {len(strategies)}")
    print(f"{'=' * 70}\n")

    for i, strategy in enumerate(strategies, 1):
        # Determine smote_ratio for this strategy
        strategy_name = strategy.value if isinstance(strategy, ImbalanceStrategy) else strategy
        if 'smote' in strategy_name and strategy != ImbalanceStrategy.SMOTE_FULL:
            ratio = smote_ratios[0] if len(smote_ratios) == 1 else smote_ratios[i % len(smote_ratios)]
        else:
            ratio = 0.5

        if verbose:
            print("=" * 70)
            print(tu.bold_and_colored_text(f"[{i}/{len(strategies)}] Testing strategy: {strategy_name}",tu.Color.BLUE))
            print("-" * 70)

        # Build fresh model for each strategy
        model = model_builder()

        # Train with strategy
        result = train_with_imbalance_handling(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy=strategy,
            smote_ratio=ratio,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            random_state=random_state,
            train_fn=train_fn,
            verbose=verbose,
            model_verbosity=model_verbosity
        )

        # Store results using string key for consistent DataFrame handling
        strategy_key = strategy.value if isinstance(strategy, ImbalanceStrategy) else strategy
        results[strategy_key] = result

        if verbose:
            # Calculate and display the metric value
            if optimize_for_str == 'recall_weighted':
                # Calculate weighted score using helper function
                weighted_score = calculate_recall_weighted(
                    result.val_metrics.roc_auc,
                    result.val_metrics.recall,
                    result.val_metrics.f1
                )
                print(f"\n   Finished Training: ✓ {optimize_for_str}: {weighted_score:.4f}")
            else:
                print(f"\n   Finished Training: ✓ {optimize_for_str}: {getattr(result.val_metrics, optimize_for_str):.4f}")

    # Create comparison DataFrame
    comparison_data = []
    for strategy_key, result in results.items():
        row = {
            'strategy': strategy_key,
            'samples': result.samples_after,
            **result.val_metrics.to_dict()
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('strategy')

    # Handle recall_weighted as a special case
    if optimize_for_str == 'recall_weighted':
        # Calculate weighted score using helper function
        comparison_df['recall_weighted'] = comparison_df.apply(
            lambda row: calculate_recall_weighted(
                row['roc_auc'],
                row['recall'],
                row['f1']
            ),
            axis=1
        )

    comparison_df = comparison_df.sort_values(by=optimize_for_str, ascending=False)

    # Identify best strategy
    best_strategy = comparison_df.index[0]
    ranking = [(idx, row[optimize_for_str]) for idx, row in comparison_df.iterrows()]

    # Create comparison object
    comparison = OptimizationComparison(
        results=results,
        comparison_df=comparison_df,
        best_strategy=best_strategy,
        best_metric=optimize_for_str,
        ranking=ranking
    )

    if verbose:
        comparison.print_comparison()

    return comparison
