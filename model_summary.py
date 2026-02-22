import model_optimizer
from feature_engineering import PreparedData

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

def generate_model_selection_summary(comparison: OptimizationComparison, best_result: ImbalanceTrainingResult,
    results: ModelEvaluationResult, data: PreparedData, imbalance_analysis: ImbalanceAnalysisResult):
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
    """
    from IPython.display import display, Markdown
    import numpy as np

    # Extract values
    best_strategy = comparison.best_strategy
    #best_threshold_info = results['best_threshold']
    #best_threshold_info = results.best_threshold
    #best_threshold = best_threshold_info['threshold']
    best_threshold = results.best_threshold

    test_auc = results['auc']
    #cm = results['confusion_matrix']
    cm = results.confusion_matrix
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

    # Best result training info
    best_val_auc = best_result.val_auc
    best_epoch = best_result.best_epoch

    # Class weights
    class_weights = best_result.class_weights if hasattr(best_result, 'class_weights') else None

    # Training sample counts
    train_counts = best_result.train_class_counts

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
        train_samples = sum(strategy_result.train_class_counts.values())
        class_dist = str(strategy_result.train_class_counts)
        val_auc = strategy_result.val_auc
        epoch = strategy_result.best_epoch
        comparison_rows.append(f"| **{strategy_name}** | {train_samples:,} | {class_dist} | {val_auc:.4f} | {epoch} |")

    comparison_table = "\n".join(comparison_rows)

    # Generate markdown
    md = f"""# Model Selection Summary: Findings and Motivations

## Executive Summary
After comprehensive experimentation with imbalance handling strategies and threshold optimization, we selected **{best_strategy}** with **early stopping on validation AUC** and **recall-weighted threshold optimization**. This configuration achieves **{recall_pct:.2f}% recall** on defaults, catching {defaults_caught} out of {total_defaults} actual defaults in the test set.

---

## 1. Imbalance Handling Strategy Selection

### The Challenge
Our dataset exhibits **{imbalance_analysis.severity}**:
- **Class 0 (Paid)**: {class_0_count:,} samples ({class_0_pct:.2f}%)
- **Class 1 (Default)**: {class_1_count:,} samples ({class_1_pct:.2f}%)
- **Imbalance Ratio**: {imbalance_ratio:.2f}:1

### Strategies Tested
We compared {len(comparison.results)} imbalance handling strategies:

| Strategy | Training Samples | Class Distribution | Validation AUC | Best Epoch |
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

    md += f"""### Why {best_strategy}?

1. **Best Validation Performance**: Achieved highest validation AUC of **{best_val_auc:.4f}**, outperforming all other strategies
2. **Optimal Training Signal**: Converged at epoch {best_epoch}

---

## 2. Early Stopping Strategy

### Monitoring Metric: Validation AUC

We chose **`monitor='val_auc'`** with **`patience=5`** because:

1. **AUC is Threshold-Independent**: Measures model's ability to rank predictions correctly
2. **Robust to Imbalance**: Unlike accuracy, AUC evaluates performance across all possible thresholds
3. **Prevents Overfitting**: Model stopped at epoch {best_epoch} when validation AUC plateaued

**Training Dynamics**:
- Best epoch: {best_epoch}
- Best val_auc: **{best_val_auc:.4f}**

---

## 3. Threshold Optimization Strategy

### Optimization Metric: RECALL_WEIGHTED

**Why Recall-Weighted Optimization?**

In loan default prediction, **missing a default (False Negative) is far more costly** than incorrectly flagging a paid loan as default (False Positive). Our business priority is to **maximize default detection** while maintaining reasonable precision.

### Selected Threshold: {best_threshold}

**Rationale**:
1. **High Recall**: Achieves **{recall_pct:.2f}% recall**, catching {defaults_caught} out of {total_defaults} defaults
2. **Business Alignment**: Prioritizes default detection over false positives

**Cost-Benefit Analysis**:
```
Assume average loan: $15,000
Assume default recovery rate: 30%

Per loan costs:
- False Negative (missed default): $15,000 × 70% = $10,500 loss
- False Positive (incorrect flag): ~$500 manual review cost

At threshold {best_threshold}:
- Missed defaults (FN): {fn} × $10,500 = ${fn * 10500:,}
- Incorrect flags (FP): {fp} × $500 = ${fp * 500:,}
- Total cost: ${(fn * 10500) + (fp * 500):,}
```

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

1. **Asymmetric Costs**: Missing a $15,000 default is **21x more expensive** than a manual review
2. **Risk Management**: In lending, conservative predictions protect against catastrophic losses
3. **Baseline Comparison**: Random chance would catch only **{baseline_catch_rate:.0f}% of defaults**; our model catches **{recall_pct:.0f}%**

---

## 5. Model Architecture

**Neural Network Configuration**:
```python
Input Layer: {data.X_train.shape[1]} features (after one-hot encoding)
Hidden Layer 1: 32 neurons, ReLU activation
Dropout 1: 30% dropout rate
Hidden Layer 2: 16 neurons, ReLU activation
Dropout 2: 30% dropout rate
Output Layer: 1 neuron, Sigmoid activation

Optimizer: Adam (lr=0.001)
Loss: Binary Crossentropy (with class weights)
```

---

## 6. Key Takeaways

### Business Impact
- **Current Model**: Catches {defaults_caught}/{total_defaults} defaults ({recall_pct:.0f}% recall at threshold {best_threshold})
- **Baseline (Random)**: Would catch ~{int(total_defaults * baseline_catch_rate / 100)}/{total_defaults} defaults ({baseline_catch_rate:.0f}% by chance)
- **Improvement**: **+{defaults_caught - int(total_defaults * baseline_catch_rate / 100)} additional defaults caught**
- **Financial Impact**: ${(defaults_caught - int(total_defaults * baseline_catch_rate / 100)) * 15000:,} in prevented losses per test batch

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
