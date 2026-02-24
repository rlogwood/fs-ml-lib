try:
    from lib.model_optimizer import OptimizationComparison, ImbalanceTrainingResult
    from lib.class_imbalance import ImbalanceAnalysisResult
    from lib.model_evaluator import ModelEvaluationResult
    from lib.feature_engineering import PreparedData
    from lib.utility import get_model_architecture_info, ModelArchitectureInfo
except ImportError:
    from model_optimizer import OptimizationComparison, ImbalanceTrainingResult
    from class_imbalance import ImbalanceAnalysisResult
    from model_evaluator import ModelEvaluationResult
    from feature_engineering import PreparedData
    from utility import get_model_architecture_info, ModelArchitectureInfo

from keras.callbacks import EarlyStopping
from collections.abc import Callable
def generate_model_selection_summary(comparison: OptimizationComparison, best_result: ImbalanceTrainingResult,
                                     model_eval_results: ModelEvaluationResult, data: PreparedData,
                                     imbalance_analysis: ImbalanceAnalysisResult,
                                     early_stop: EarlyStopping,
                                     cost_benefit_fn: Callable[[float, float, float, float, float], str],
                                     monitoring_explanation: Callable[[EarlyStopping, int], str],
                                     trade_off_discussion: Callable[[int, float], str],
                                     business_impact: Callable[[int, int, float, int, float], str]):
    #= lambda x: f"**{x.monitor}** is used to monitor training performance"):
    """
    Generate a comprehensive model selection summary with actual calculated values.

    Parameters:
    -----------
    comparison : OptimizationComparison
        Results from optimize_imbalance_strategy
    best_result : ImbalanceTrainingResult
        The best performing strategy result
    results : dict
        Dictionary containing evaluation results (from evaluate_model_comprehensive)
    data : PreparedData
        The prepared data object with train/val/test splits
    result_obj : ImbalanceAnalysisResult
        Class imbalance analysis result
    cost_benefit_fn : callable, optional
        Function that takes (threshold, fn, fp, tp, tn) and returns a formatted string
        for cost-benefit analysis. If None, uses default loan default cost analysis.
    """
    from IPython.display import display, Markdown
    import numpy as np

    def readable_class_dist(class_dist: dict) -> str:
        return ", ".join(f"({k}:{v:,})" for k, v in class_dist.items())
        # description = ""
        # separator = ""
        # for k, v in class_dist.items():
        #     description += separator + f"({k}:{v:,})"
        #     separator = ", "
        # return description

    # Extract values
    best_strategy = comparison.best_strategy
    #best_threshold_info = results['best_threshold']
    #best_threshold_info = results.best_threshold
    #best_threshold = best_threshold_info['threshold']
    #best_threshold =

    test_auc = model_eval_results.auc
    cm = model_eval_results.confusion_matrix
    tn, fp, fn, tp = cm.ravel()

    # Convert numpy types to Python native types for formatting
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)

    # Calculate class distribution
    # total_samples = result_obj.total_samples
    #total_samples = result_obj.samples_after

    # result = ImbalanceAnalysisResult(
    #     n_classes=n_classes,
    #     total_samples=int(total),
    #     majority_class=class_labels[majority_idx],
    #     majority_count=int(majority_count),
    #     minority_class=class_labels[minority_idx],
    #     minority_count=int(minority_count),
    #     minority_percentage=f"{(minority_count / total) * 100:.2f}%",
    #     imbalance_ratio=f"{ratio:.2f}:1",
    #     severity=severity,
    #     recommended_action=action,
    #     class_analysis=class_analysis
    # )


    total_samples = imbalance_analysis.total_samples
    # counts = list(result_obj.class_dist_after.values())
    # class_0_count = counts[0]
    # class_1_count = counts[1]
    # class_0_pct = (class_0_count / total_samples) * 100
    # class_1_pct = (class_1_count / total_samples) * 100

    class_0_count = imbalance_analysis.majority_count
    class_1_count = imbalance_analysis.minority_count
    class_0_pct = (class_0_count / total_samples) * 100
    class_1_pct = (class_1_count / total_samples) * 100

    #imbalance_ratio = result_obj.imbalance_ratio
    imbalance_ratio = imbalance_analysis.imbalance_ratio
    best_threshold = model_eval_results.best_threshold

    # Best result training info
    # Get validation AUC from history (Keras History object has .history dict)
    if hasattr(best_result.history, 'history') and 'val_auc' in best_result.history.history:
        val_auc_list = best_result.history.history['val_auc']
        best_val_auc = max(val_auc_list)
        best_epoch = val_auc_list.index(best_val_auc) + 1  # +1 because epochs are 1-indexed
    else:
        # Fallback to val_metrics if history not available
        best_val_auc = best_result.val_metrics.roc_auc
        best_epoch = len(best_result.history.epoch) if hasattr(best_result.history, 'epoch') else 'N/A'

    # Class weights
    class_weights = best_result.class_weight_dict

    # Training sample counts (use class distribution after processing)
    train_counts = best_result.class_dist_after

    # Calculate metrics
    total_defaults = fn + tp
    defaults_caught = tp
    recall_pct = (defaults_caught / total_defaults) * 100 if total_defaults > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    baseline_catch_rate = class_1_pct

    best_model = comparison.get_best_model()
    # Get architecture info
    arch_info = get_model_architecture_info(best_model)
    # Use as string
    print(f"Best model: {arch_info}")
    # Output: "RandomForestClassifier (tree_ensemble), n_estimators=100, max_depth=10"
    # Access structured data
    print(f"Model family: {arch_info.model_family}")  # "tree_ensemble"
    print(f"Config: {arch_info.config}")  # {'n_estimators': 100, 'max_depth': 10, ...}

    # Build comparison table
    comparison_rows = []
    for strategy_name, strategy_result in comparison.results.items():
        train_samples = sum(strategy_result.class_dist_after.values())
        #class_dist = str(strategy_result.class_dist_after)
        distribution = readable_class_dist(strategy_result.class_dist_after)
        #val_auc = strategy_result.val_metrics.roc_auc
        # print("=" * 70)
        # print(f"strategy_result.history: {strategy_result.history}")
        # print("=" * 70)
        # print(f"strategy_result.history.history: {strategy_result.history.history}")
        # print("=" * 70)
        # print(f"early_stop.monitor: {early_stop.monitor}")
        # print("=" * 70)
        # Get best epoch from history if available
        if hasattr(strategy_result.history, 'history') and early_stop.monitor in strategy_result.history.history:
            monitor_list = strategy_result.history.history[early_stop.monitor]
            max_monitor_val = max(monitor_list)
            epoch = monitor_list.index(max_monitor_val) + 1
        else:
            epoch = len(strategy_result.history.epoch) if hasattr(strategy_result.history, 'epoch') else 'N/A'
            max_monitor_val = np.nan

        comparison_rows.append(f"| **{strategy_name}** | {train_samples:,} | {distribution} | {max_monitor_val:.4f} | {epoch} |")

    comparison_table = "\n".join(comparison_rows)

    summary_lines = []
    best_model.summary(print_fn=lambda x: summary_lines.append(x))
    model_summary_str = '\n'.join(summary_lines)

    print(f"Model summary:\n{model_summary_str}")

    #TODO: FIX ME! hard coded class names, capped at 2 classes (see below)
    # Generate markdown
    md = f"""# Model Selection Summary: Findings and Motivations

## Executive Summary
After comprehensive experimentation with imbalance handling strategies and threshold optimization, we selected **{best_strategy}** with **early stopping on validation AUC** and **recall-weighted threshold optimization**. This configuration achieves **{recall_pct:.2f}% recall** on defaults, catching {defaults_caught} out of {total_defaults} actual defaults in the test set.

---

## 1. Imbalance Handling Strategy Selection

### The Challenge
Our dataset exhibits **{imbalance_analysis.severity}**:
#### TODO: FIX ME! hard coded class names, capped at 2 classes
- **Class 0 (Paid)**: {class_0_count:,} samples ({class_0_pct:.2f}%)
- **Class 1 (Default)**: {class_1_count:,} samples ({class_1_pct:.2f}%)
- **Imbalance Ratio**: {imbalance_ratio}

### Strategies Tested
We compared {len(comparison.results)} imbalance handling strategies:

| Strategy | Training Samples | Class Distribution | Validation {early_stop.monitor} | Best Epoch |
|----------|------------------|-------------------|----------------|------------|
{comparison_table}

"""

    # Add class weights section if available
    if class_weights:
        weight_0 = class_weights.get(0, 'N/A')
        weight_1 = class_weights.get(1, 'N/A')
        if isinstance(weight_0, (int, float)) and isinstance(weight_1, (int, float)):
            md += f"""**Calculated Class Weights**: 
- Class 0 (Paid): {weight_0:.3f}
- Class 1 (Default): {weight_1:.3f}

"""
    #best_strategy_safe = best_strategy.replace('%', '%%') if best_strategy else ''
    md += f"""### Why {best_strategy}?

1. **Best Validation Performance**: Achieved highest validation {early_stop.monitor} of **{best_val_auc:.4f}**, outperforming all other strategies
2. **Optimal Training Signal**: Converged at epoch {best_epoch}

---

## 2. Early Stopping Strategy

### Monitoring Metric: Validation AUC

{monitoring_explanation(early_stop, best_epoch)}

**Training Dynamics**:
- Best epoch: {best_epoch}
- Best val_auc: **{best_val_auc:.4f}**

---

## 3. Threshold Optimization Strategy

### Optimization Metric: {comparison.best_metric}

**Why Recall-Weighted Optimization?**

In loan default prediction, **missing a default (False Negative) is far more costly** than incorrectly flagging a paid loan as default (False Positive). Our business priority is to **maximize default detection** while maintaining reasonable precision.

### Selected Threshold: {best_threshold}

**Rationale**:
1. **High Recall**: Achieves **{recall_pct:.2f}% recall**, catching {defaults_caught} out of {total_defaults} defaults
2. **Business Alignment**: Prioritizes default detection over false positives

{cost_benefit_fn(best_threshold, fn, fp, tp, tn)}

---

## 4. Final Model Performance

### Test Set Results (Threshold = {best_threshold})

**Confusion Matrix**:
```
                Predicted
                Paid    Default
Actual  Paid     {tn:,}      {fp:,}
        Default   {fn}        {tp}
```

**Breakdown**:
- **True Negatives (TN)**: {tn:,} ({(tn / (tn + fp) * 100):.1f}% of paid loans correctly identified)
- **False Positives (FP)**: {fp:,} ({(fp / (tn + fp) * 100):.1f}% of paid loans flagged for review)
- **False Negatives (FN)**: {fn} ({(fn / (fn + tp) * 100):.1f}% missed defaults - CRITICAL METRIC)
- **True Positives (TP)**: {tp} ({(tp / (fn + tp) * 100):.1f}% defaults caught)

**Key Metrics**:
- **Test AUC-ROC**: {test_auc:.4f}
- **Recall (Default Class)**: {recall_pct:.2f}%
- **Precision (Default Class)**: {precision * 100:.2f}%

### Why This Trade-off Makes Sense
{trade_off_discussion(baseline_catch_rate, recall_pct)}

---

## 5. Model Architecture



**Neural Network Configuration**:
```
{model_summary_str}
```
---

## 6. Key Takeaways

{business_impact(defaults_caught, total_defaults, recall_pct, baseline_catch_rate, best_threshold)}
---

## Next Steps

1. **Feature Engineering**: Create interaction terms, risk scores, and temporal features
2. **Alternative Models**: Compare against Random Forest, XGBoost, and Gradient Boosting
3. **Ensemble Methods**: Combine multiple models for improved robustness
4. **Hyperparameter Tuning**: Optimize learning rate, network depth, and dropout rates

Our goal is to improve **AUC-ROC beyond {test_auc:.4f}** while maintaining **high recall (>95%)**.
"""

    display(Markdown(md))
    return md

# Generate and display the summary
# NOTE: All required variables should exist from previous cells:
#   - comparison: from cell 34
#   - best_result: from cell 34
#   - results: from cell 35 (MUST RUN CELL 35 FIRST!)
#   - data: from cell 30
#   - result: from cell 19
