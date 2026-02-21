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


@dataclass
class ImbalanceTrainingResult:
    """Results from training with imbalance handling"""
    strategy: Union[ImbalanceStrategy, str]
    history: Any
    model: Any
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
    val_metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self):
        """Print a summary of the training configuration"""
        strategy_name = self.strategy.value if isinstance(self.strategy, ImbalanceStrategy) else self.strategy
        print(f"\n{'=' * 70}")
        print(f"IMBALANCE HANDLING SUMMARY: {strategy_name}")
        print(f"{'=' * 70}")
        print(f"Samples: {self.samples_before:,} → {self.samples_after:,}")
        print(f"\nClass Distribution Before:")
        for cls, count in self.class_dist_before.items():
            pct = count / self.samples_before * 100
            print(f"  Class {cls}: {count:,} ({pct:.1f}%)")
        print(f"\nClass Distribution After:")
        for cls, count in self.class_dist_after.items():
            pct = count / self.samples_after * 100
            print(f"  Class {cls}: {count:,} ({pct:.1f}%)")
        if self.class_weight_dict:
            print(f"\nClass Weights Applied:")
            for cls, weight in self.class_weight_dict.items():
                print(f"  Class {cls}: {weight:.4f}")
        if self.smote_ratio:
            print(f"\nSMOTE Ratio: {self.smote_ratio}")

        if self.val_metrics:
            print(f"\nValidation Metrics:")
            for metric, value in self.val_metrics.items():
                print(f"  {metric}: {value:.4f}")
        print(f"{'=' * 70}\n")


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
        callbacks=None,
        random_state=42,
        train_fn=None,
        verbose=True
):
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

    # Calculate class weights if needed
    if auto_calculate_weights and class_weight_dict is None:
        if strategy in [ImbalanceStrategy.CLASS_WEIGHTS, ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS]:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes.astype(int), weights))
            if verbose:
                print(f"\n📊 Auto-calculated class weights: {class_weight_dict}")

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
        from lib.model_trainer import train_model_with_class_weights
        history = train_model_with_class_weights(
            model, X_train_final, y_train_final, X_val, y_val,
            weights, epochs=epochs, callbacks=callbacks
        )
    else:
        history = train_fn(
            model, X_train_final, y_train_final, X_val, y_val,
            weights, epochs, callbacks
        )

    # Calculate validation metrics
    y_val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    y_val_pred_proba = model.predict(X_val, verbose=0).flatten()

    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, zero_division=0),
        'recall': recall_score(y_val, y_val_pred, zero_division=0),
        'f1': f1_score(y_val, y_val_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_val_pred_proba)
    }

    # Create result object
    result = ImbalanceTrainingResult(
        strategy=strategy,
        history=history,
        model=model,
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
        optimize_for='f1',
        epochs=50,
        callbacks=None,
        random_state=42,
        verbose=True
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
    optimize_for : str
        Metric to optimize ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
    epochs : int
        Training epochs per strategy
    callbacks : list, optional
        Keras callbacks
    random_state : int
        Random seed
    verbose : bool
        Print progress

    Returns:
    --------
    OptimizationComparison
        Comparison results with best strategy identified
    """

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
    print(f"   Optimizing for: {optimize_for}")
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
            print(f"\n[{i}/{len(strategies)}] Testing strategy: {strategy_name}")
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
            callbacks=callbacks,
            random_state=random_state,
            verbose=verbose
        )

        # Store results using string key for consistent DataFrame handling
        strategy_key = strategy.value if isinstance(strategy, ImbalanceStrategy) else strategy
        results[strategy_key] = result

        if verbose:
            print(f"   ✓ {optimize_for}: {result.val_metrics[optimize_for]:.4f}")

    # Create comparison DataFrame
    comparison_data = []
    for strategy_key, result in results.items():
        row = {
            'strategy': strategy_key,
            'samples': result.samples_after,
            **result.val_metrics
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('strategy')
    comparison_df = comparison_df.sort_values(by=optimize_for, ascending=False)

    # Identify best strategy
    best_strategy = comparison_df.index[0]
    ranking = [(idx, row[optimize_for]) for idx, row in comparison_df.iterrows()]

    # Create comparison object
    comparison = OptimizationComparison(
        results=results,
        comparison_df=comparison_df,
        best_strategy=best_strategy,
        best_metric=optimize_for,
        ranking=ranking
    )

    if verbose:
        comparison.print_comparison()

    return comparison
