"""
model_comparison.py - Comprehensive model comparison framework

Provides a unified interface for training, evaluating, and comparing different
types of classification models (Neural Networks, Random Forest, XGBoost, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time


class ModelType(Enum):
    """Supported model types"""
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_type: ModelType
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    class_weight: Optional[Union[str, Dict]] = None

    def __post_init__(self):
        """Set default parameters if not provided"""
        if not self.params:
            self.params = self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for each model type"""
        defaults = {
            ModelType.NEURAL_NETWORK: {
                'layers': [32, 16],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'epochs': 50
            },
            ModelType.RANDOM_FOREST: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 50,
                'min_samples_leaf': 20,
                'random_state': 42,
                'n_jobs': -1
            },
            ModelType.XGBOOST: {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            },
            ModelType.LOGISTIC_REGRESSION: {
                'max_iter': 1000,
                'random_state': 42
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
        return defaults.get(self.model_type, {})


@dataclass
class ModelResult:
    """Results from a single model training and evaluation"""
    model_type: ModelType
    name: str
    model: Any
    config: ModelConfig

    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    # Predictions
    y_pred: np.ndarray
    y_pred_proba: np.ndarray

    # Confusion matrix
    confusion_matrix: np.ndarray

    # Training metadata
    training_time: float
    n_features: int
    n_samples_train: int
    n_samples_test: int

    # Additional data
    classification_report: str = ""
    feature_importance: Optional[pd.DataFrame] = None
    history: Optional[Any] = None  # For neural networks

    def summary(self):
        """Print a summary of model results"""
        print(f"\n{'='*70}")
        print(f"MODEL: {self.name} ({self.model_type.value})")
        print(f"{'='*70}")
        print(f"Training samples: {self.n_samples_train:,}")
        print(f"Test samples: {self.n_samples_test:,}")
        print(f"Features: {self.n_features}")
        print(f"Training time: {self.training_time:.2f}s")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1-Score:  {self.f1:.4f}")
        print(f"  ROC-AUC:   {self.roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(self.confusion_matrix)
        if self.classification_report:
            print(f"\nClassification Report:")
            print(self.classification_report)
        print(f"{'='*70}\n")


@dataclass
class ComparisonInput:
    """Input data for model comparison"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=lambda: ['Class 0', 'Class 1'])


@dataclass
class ComparisonResult:
    """Results from comparing multiple models"""
    results: Dict[str, ModelResult]
    comparison_df: pd.DataFrame
    best_model_name: str
    best_metric: str
    ranking: List[tuple]

    def print_comparison(self):
        """Print formatted comparison table"""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"Optimized for: {self.best_metric}")
        print(f"\n{self.comparison_df.to_string(index=False)}")
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        best_result = self.results[self.best_model_name]
        print(f"   {self.best_metric.upper()}: {getattr(best_result, self.best_metric):.4f}")
        print(f"   Training Time: {best_result.training_time:.2f}s")
        print(f"{'='*80}\n")

    def get_best_model(self):
        """Return the best model"""
        return self.results[self.best_model_name].model

    def get_best_result(self) -> ModelResult:
        """Return the best model result"""
        return self.results[self.best_model_name]


def build_model(config: ModelConfig, input_dim: int, scale_pos_weight: Optional[float] = None):
    """
    Build a model based on configuration

    Parameters:
    -----------
    config : ModelConfig
        Model configuration
    input_dim : int
        Number of input features
    scale_pos_weight : float, optional
        For imbalanced datasets (XGBoost)

    Returns:
    --------
    model : Any
        Initialized model
    """
    try:
        from lib.model_trainer import build_neural_network
    except ImportError:
        from model_trainer import build_neural_network

    if config.model_type == ModelType.NEURAL_NETWORK:
        return build_neural_network(
            input_dim=input_dim,
            layers=config.params.get('layers', [32, 16]),
            dropout_rate=config.params.get('dropout_rate', 0.3),
            learning_rate=config.params.get('learning_rate', 0.001)
        )

    elif config.model_type == ModelType.RANDOM_FOREST:
        params = config.params.copy()
        if config.class_weight:
            params['class_weight'] = config.class_weight
        return RandomForestClassifier(**params)

    elif config.model_type == ModelType.XGBOOST:
        try:
            import xgboost as xgb
            params = config.params.copy()
            if scale_pos_weight:
                params['scale_pos_weight'] = scale_pos_weight
            return xgb.XGBClassifier(**params)
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    elif config.model_type == ModelType.LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression
        params = config.params.copy()
        if config.class_weight:
            params['class_weight'] = config.class_weight
        return LogisticRegression(**params)

    elif config.model_type == ModelType.GRADIENT_BOOSTING:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**config.params)

    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def train_and_evaluate_model(
    config: ModelConfig,
    data: ComparisonInput,
    verbose: bool = True
) -> ModelResult:
    """
    Train and evaluate a single model

    Parameters:
    -----------
    config : ModelConfig
        Model configuration
    data : ComparisonInput
        Training and test data
    verbose : bool
        Print progress

    Returns:
    --------
    ModelResult
        Training and evaluation results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {config.name} ({config.model_type.value})...")
        print(f"{'='*70}")

    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = None
    if config.model_type == ModelType.XGBOOST:
        scale_pos_weight = (data.y_train == 0).sum() / (data.y_train == 1).sum()

    # Build model
    model = build_model(config, data.X_train.shape[1], scale_pos_weight)

    # Train model
    start_time = time.time()

    if config.model_type == ModelType.NEURAL_NETWORK:
        # Neural network training
        try:
            from lib.model_trainer import train_model_with_class_weights
        except ImportError:
            from model_trainer import train_model_with_class_weights

        # Use validation data if provided
        X_val = data.X_val if data.X_val is not None else data.X_test
        y_val = data.y_val if data.y_val is not None else data.y_test

        # Calculate class weights if specified
        class_weights = None
        if config.class_weight == 'balanced':
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(data.y_train)
            weights = compute_class_weight('balanced', classes=classes, y=data.y_train)
            class_weights = dict(zip(classes.astype(int), weights))

        history = train_model_with_class_weights(
            model, data.X_train, data.y_train, X_val, y_val,
            class_weights,
            epochs=config.params.get('epochs', 50),
            callbacks=None
        )
        history_obj = history
    else:
        # Scikit-learn style training
        model.fit(data.X_train, data.y_train)
        history_obj = None

    training_time = time.time() - start_time

    # Make predictions
    if config.model_type == ModelType.NEURAL_NETWORK:
        y_pred_proba = model.predict(data.X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(data.X_test)
        y_pred_proba = model.predict_proba(data.X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(data.y_test, y_pred)
    precision = precision_score(data.y_test, y_pred, zero_division=0)
    recall = recall_score(data.y_test, y_pred, zero_division=0)
    f1 = f1_score(data.y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(data.y_test, y_pred_proba)
    cm = confusion_matrix(data.y_test, y_pred)

    # Classification report
    report = classification_report(
        data.y_test, y_pred,
        target_names=data.class_names,
        zero_division=0
    )

    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': range(data.X_train.shape[1]),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    if verbose:
        print(f"✓ Training complete in {training_time:.2f}s")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Recall:  {recall:.4f}")
        print(f"  F1:      {f1:.4f}")

    # Create result
    result = ModelResult(
        model_type=config.model_type,
        name=config.name,
        model=model,
        config=config,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        confusion_matrix=cm,
        training_time=training_time,
        n_features=data.X_train.shape[1],
        n_samples_train=len(data.X_train),
        n_samples_test=len(data.X_test),
        classification_report=report,
        feature_importance=feature_importance,
        history=history_obj
    )

    return result


def compare_models(
    configs: List[ModelConfig],
    data: ComparisonInput,
    optimize_for: str = 'f1',
    verbose: bool = True
) -> ComparisonResult:
    """
    Train and compare multiple models

    Parameters:
    -----------
    configs : List[ModelConfig]
        List of model configurations to compare
    data : ComparisonInput
        Training and test data
    optimize_for : str
        Metric to optimize ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
    verbose : bool
        Print progress

    Returns:
    --------
    ComparisonResult
        Comparison results with best model identified
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔬 MODEL COMPARISON")
        print(f"   Models to compare: {len(configs)}")
        print(f"   Optimizing for: {optimize_for}")
        print(f"{'='*80}")

    results = {}

    # Train each model
    for i, config in enumerate(configs, 1):
        if verbose:
            print(f"\n[{i}/{len(configs)}]")

        result = train_and_evaluate_model(config, data, verbose=verbose)
        results[config.name] = result

    # Create comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        row = {
            'Model': name,
            'Type': result.model_type.value,
            'accuracy': result.accuracy,
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'roc_auc': result.roc_auc,
            'train_time_s': result.training_time
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(by=optimize_for, ascending=False)
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))

    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    ranking = [(row['Model'], row[optimize_for]) for _, row in comparison_df.iterrows()]

    # Create comparison result
    comparison = ComparisonResult(
        results=results,
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_metric=optimize_for,
        ranking=ranking
    )

    if verbose:
        comparison.print_comparison()

    return comparison


def create_default_configs(
    use_neural_network: bool = True,
    use_random_forest: bool = True,
    use_xgboost: bool = True,
    use_logistic: bool = False,
    use_gradient_boosting: bool = False,
    class_weight: str = 'balanced'
) -> List[ModelConfig]:
    """
    Create default model configurations

    Parameters:
    -----------
    use_* : bool
        Which models to include
    class_weight : str
        How to handle class imbalance ('balanced' or None)

    Returns:
    --------
    List[ModelConfig]
        List of model configurations
    """
    configs = []

    if use_neural_network:
        configs.append(ModelConfig(
            model_type=ModelType.NEURAL_NETWORK,
            name="Neural Network",
            class_weight=class_weight
        ))

    if use_random_forest:
        configs.append(ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            name="Random Forest",
            class_weight=class_weight
        ))

    if use_xgboost:
        try:
            import xgboost
            configs.append(ModelConfig(
                model_type=ModelType.XGBOOST,
                name="XGBoost",
                class_weight=class_weight
            ))
        except ImportError:
            if use_xgboost:
                print("⚠️  XGBoost not installed, skipping")

    if use_logistic:
        configs.append(ModelConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            name="Logistic Regression",
            class_weight=class_weight
        ))

    if use_gradient_boosting:
        configs.append(ModelConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            name="Gradient Boosting",
            class_weight=class_weight
        ))

    return configs
