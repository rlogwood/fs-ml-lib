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
                                     business_impact: Callable[[int, int, float, int, float], str],
                                     executive_summary: Callable[[], str],
                                     class_labels: dict = None):
    #= lambda x: f"**{x.monitor}** is used to monitor training performance"):
    """
    Generate a comprehensive model selection summary with actual calculated values.

    Parameters:
    -----------
    comparison : OptimizationComparison
        Results from optimize_imbalance_strategy
    best_result : ImbalanceTrainingResult
        The best performing strategy result
    model_eval_results : ModelEvaluationResult
        Model evaluation results from evaluate_model_comprehensive
    data : PreparedData
        The prepared data object with train/val/test splits
    imbalance_analysis : ImbalanceAnalysisResult
        Class imbalance analysis result
    early_stop : EarlyStopping
        Early stopping callback used during training
    cost_benefit_fn : callable
        Function that takes (threshold, fn, fp, tp, tn) and returns a formatted string
        for cost-benefit analysis
    monitoring_explanation : callable
        Function that takes (early_stop, best_epoch) and returns explanation text
    trade_off_discussion : callable
        Function that takes (baseline_catch_rate, recall_pct) and returns discussion text
    business_impact : callable
        Function that takes (defaults_caught, total_defaults, recall_pct, baseline_catch_rate, best_threshold)
        and returns business impact text
    class_labels : dict, optional
        Mapping of class values to human-readable labels {class_value: "Display Name"}
        Example: {0: "Paid", 1: "Default"}
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
    tn, fp, fn, tp = cm.flatten().tolist()

    # Convert numpy types to Python native types for formatting
    # tn = int(tn)
    # fp = int(fp)
    # fn = int(fn)
    # tp = int(tp)

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

    # Training sample counts (use class distribution after processing)
    train_counts = best_result.class_dist_after

    # Calculate metrics
    total_defaults = fn + tp
    defaults_caught = tp
    recall_pct = (defaults_caught / total_defaults) * 100 if total_defaults > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    baseline_catch_rate = (imbalance_analysis.minority_count / imbalance_analysis.total_samples) * 100

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

    # Generate markdown
    md = f"""# Model Selection Summary: Findings and Motivations

{executive_summary}
---

## 1. Imbalance Handling Strategy Selection

### The Challenge
{imbalance_analysis.display_markdown()}

### Strategies Tested
We compared {len(comparison.results)} imbalance handling strategies:

| Strategy | Training Samples | Class Distribution | Validation {early_stop.monitor} | Best Epoch |
|----------|------------------|-------------------|----------------|------------|
{comparison_table}

"""
    # Add class weights section if available
    md += best_result.display_markdown(class_labels=class_labels)
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
