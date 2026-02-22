"""
Pytest tests for model_evaluator module
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
import model_evaluator as me
import model_optimizer as mo


@pytest.fixture
def sample_predictions():
    """Create sample predictions"""
    np.random.seed(42)
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.array([0] * 70 + [1] * 10 + [0] * 15 + [1] * 5)
    y_pred_proba = np.random.rand(100)
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def realistic_predictions():
    """Create realistic predictions with signal"""
    np.random.seed(42)
    n_samples = 200
    y_true = np.array([0] * 160 + [1] * 40)

    # Generate predictions with some signal
    y_pred_proba = np.random.rand(n_samples)
    y_pred_proba[160:] += 0.3  # Minority class more likely to have higher prob
    y_pred_proba = np.clip(y_pred_proba, 0, 1)

    y_pred = (y_pred_proba >= 0.5).astype(int)

    return y_true, y_pred, y_pred_proba


@pytest.fixture
def mock_history():
    """Create mock training history"""
    history = MagicMock()
    history.history = {
        'loss': [0.7, 0.6, 0.5, 0.4],
        'accuracy': [0.6, 0.7, 0.75, 0.8],
        'precision': [0.5, 0.6, 0.65, 0.7],
        'recall': [0.4, 0.5, 0.6, 0.65],
        'val_loss': [0.75, 0.65, 0.55, 0.45],
        'val_accuracy': [0.55, 0.65, 0.7, 0.75],
        'val_precision': [0.45, 0.55, 0.6, 0.65],
        'val_recall': [0.35, 0.45, 0.55, 0.6]
    }
    return history


class TestConfusionMatrix:
    """Tests for confusion matrix plotting"""

    def test_plot_confusion_matrix(self, sample_predictions):
        """Test confusion matrix plotting"""
        y_true, y_pred, _ = sample_predictions
        cm, fig = me.plot_confusion_matrix(y_true, y_pred)

        assert cm is not None
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2 subplots + 2 colorbars

        plt.close(fig)

    def test_plot_confusion_matrix_custom_labels(self, sample_predictions):
        """Test confusion matrix with custom labels"""
        y_true, y_pred, _ = sample_predictions
        cm, fig = me.plot_confusion_matrix(
            y_true,
            y_pred,
            labels=['Good', 'Bad']
        )

        assert cm is not None
        assert fig is not None
        plt.close(fig)


class TestClassificationMetrics:
    """Tests for classification metrics"""

    def test_print_classification_metrics_without_proba(self, sample_predictions, capsys):
        """Test printing metrics without probabilities"""
        y_true, y_pred, _ = sample_predictions

        # Should not raise any errors
        me.print_classification_metrics(y_true, y_pred)
        captured = capsys.readouterr()

        # Check that some output was produced
        assert len(captured.out) > 0

    def test_print_classification_metrics_with_proba(self, sample_predictions, capsys):
        """Test printing metrics with probabilities"""
        y_true, y_pred, y_pred_proba = sample_predictions

        # Should not raise any errors
        me.print_classification_metrics(y_true, y_pred, y_pred_proba)
        captured = capsys.readouterr()

        # Check that AUC is mentioned when proba provided
        assert 'AUC' in captured.out or 'auc' in captured.out.lower()


class TestROCCurve:
    """Tests for ROC curve plotting"""

    def test_plot_roc_curve(self, sample_predictions):
        """Test ROC curve plotting"""
        y_true, _, y_pred_proba = sample_predictions
        fig, auc_score = me.plot_roc_curve(y_true, y_pred_proba)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(auc_score, float)
        assert 0.0 <= auc_score <= 1.0

        plt.close(fig)

    def test_plot_roc_curve_perfect_classifier(self):
        """Test ROC curve with perfect predictions"""
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred_proba = np.array([0.1] * 50 + [0.9] * 50)

        fig, auc_score = me.plot_roc_curve(y_true, y_pred_proba)

        assert abs(auc_score - 1.0) < 0.1
        plt.close(fig)


class TestPrecisionRecallCurve:
    """Tests for precision-recall curve plotting"""

    def test_plot_precision_recall_curve(self, sample_predictions):
        """Test precision-recall curve plotting"""
        y_true, _, y_pred_proba = sample_predictions
        fig = me.plot_precision_recall_curve(y_true, y_pred_proba)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)


class TestThresholdOptimization:
    """Tests for threshold optimization"""

    def test_optimize_threshold_default(self, sample_predictions):
        """Test threshold optimization with defaults"""
        y_true, _, y_pred_proba = sample_predictions

        best_result, results_df = me.optimize_threshold(
            y_true,
            y_pred_proba,
            verbose=False
        )

        assert isinstance(best_result, dict)
        assert isinstance(results_df, pd.DataFrame)

        # Check required keys
        assert 'threshold' in best_result
        assert 'accuracy' in best_result
        assert 'precision' in best_result
        assert 'recall' in best_result
        assert 'f1' in best_result

        # Check DataFrame
        assert len(results_df) == 5  # Default 5 thresholds
        assert 'threshold' in results_df.columns

    def test_optimize_threshold_custom_thresholds(self, sample_predictions):
        """Test threshold optimization with custom thresholds"""
        y_true, _, y_pred_proba = sample_predictions
        custom_thresholds = [0.2, 0.4, 0.6, 0.8]

        best_result, results_df = me.optimize_threshold(
            y_true,
            y_pred_proba,
            thresholds_to_test=custom_thresholds,
            verbose=False
        )

        assert len(results_df) == 4

    def test_optimize_threshold_by_precision(self, sample_predictions):
        """Test threshold optimization prioritizing precision"""
        y_true, _, y_pred_proba = sample_predictions

        best_result, results_df = me.optimize_threshold(
            y_true,
            y_pred_proba,
            metric=mo.OptimizationMetric.PRECISION,
            verbose=False
        )

        # Best result should have highest precision in results
        max_precision_idx = results_df['precision'].idxmax()
        assert best_result['precision'] == results_df.iloc[max_precision_idx]['precision']

    def test_optimize_threshold_by_recall(self, sample_predictions):
        """Test threshold optimization prioritizing recall"""
        y_true, _, y_pred_proba = sample_predictions

        best_result, results_df = me.optimize_threshold(
            y_true,
            y_pred_proba,
            metric=mo.OptimizationMetric.RECALL,
            verbose=False
        )

        # Best result should have highest recall in results
        max_recall_idx = results_df['recall'].idxmax()
        assert best_result['recall'] == results_df.iloc[max_recall_idx]['recall']

    def test_optimize_threshold_metrics_range(self, sample_predictions):
        """Test that all metrics are in valid range"""
        y_true, _, y_pred_proba = sample_predictions

        _, results_df = me.optimize_threshold(
            y_true,
            y_pred_proba,
            verbose=False
        )

        # All metrics should be between 0 and 1
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            assert (results_df[metric] >= 0).all()
            assert (results_df[metric] <= 1).all()


class TestTrainingHistory:
    """Tests for training history visualization"""

    def test_plot_training_history(self, mock_history):
        """Test training history plotting"""
        fig = me.plot_training_history(mock_history)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # At least 4 subplots (may include legend axes)

        plt.close(fig)

    def test_plot_training_history_custom_metrics(self, mock_history):
        """Test plotting with custom metrics"""
        fig = me.plot_training_history(
            mock_history,
            metrics=['loss', 'accuracy']
        )

        assert fig is not None
        plt.close(fig)


class TestComprehensiveEvaluation:
    """Tests for comprehensive model evaluation"""

    def test_evaluate_model_comprehensive(self, sample_predictions):
        """Test comprehensive evaluation"""
        y_true, _, _ = sample_predictions
        X_test = np.random.randn(len(y_true), 10)

        # Create mock model
        mock_model = MagicMock()
        np.random.seed(42)
        mock_model.predict.return_value = np.random.rand(len(y_true), 1)
        # Delete predict_proba to ensure hasattr returns False
        del mock_model.predict_proba

        results = me.evaluate_model_comprehensive(
            mock_model,
            X_test,
            y_true
        )

        # Check that result is a dataclass
        assert hasattr(results, 'cm_fig')
        assert hasattr(results, 'roc_fig')
        assert hasattr(results, 'pr_fig')
        assert hasattr(results, 'auc')
        assert hasattr(results, 'best_threshold')
        assert hasattr(results, 'threshold_df')

        # Check types
        assert isinstance(results.cm_fig, plt.Figure)
        assert isinstance(results.roc_fig, plt.Figure)
        assert isinstance(results.pr_fig, plt.Figure)
        assert isinstance(results.auc, float)
        assert isinstance(results.best_threshold, float)
        assert isinstance(results.threshold_df, pd.DataFrame)

        # Clean up
        plt.close(results.cm_fig)
        plt.close(results.roc_fig)
        plt.close(results.pr_fig)


class TestClassDistributionAnalysis:
    """Tests for class distribution analysis"""

    def test_plot_class_distribution_analysis(self):
        """Test class distribution plotting"""
        df = pd.DataFrame({
            'target': [0] * 80 + [1] * 20
        })

        fig = me.plot_class_distribution_analysis(
            df,
            'target',
            ['Class 0', 'Class 1']
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_class_distribution_imbalanced(self):
        """Test with severely imbalanced data"""
        df = pd.DataFrame({
            'target': [0] * 95 + [1] * 5
        })

        fig = me.plot_class_distribution_analysis(
            df,
            'target',
            ['Majority', 'Minority']
        )


        assert fig is not None
        plt.close(fig)


class TestIntegration:
    """Integration tests for model evaluator"""

    def test_full_evaluation_pipeline(self, realistic_predictions):
        """Test complete evaluation pipeline"""
        y_true, y_pred, y_pred_proba = realistic_predictions

        # Test all evaluation functions
        # Confusion matrix
        cm, cm_fig = me.plot_confusion_matrix(y_true, y_pred)
        assert cm is not None
        assert cm_fig is not None
        plt.close(cm_fig)

        # ROC curve
        roc_fig, auc_score = me.plot_roc_curve(y_true, y_pred_proba)
        assert roc_fig is not None
        assert auc_score > 0
        plt.close(roc_fig)

        # PR curve
        pr_fig = me.plot_precision_recall_curve(y_true, y_pred_proba)
        assert pr_fig is not None
        plt.close(pr_fig)

        # Threshold optimization
        best_result, results_df = me.optimize_threshold(
            y_true,
            y_pred_proba,
            verbose=False
        )
        assert best_result is not None
        assert results_df is not None
