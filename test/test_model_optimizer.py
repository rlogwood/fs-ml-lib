"""
Pytest tests for model_optimizer module
"""

import pytest
import numpy as np
import pandas as pd
import model_optimizer as mo
from unittest.mock import Mock, patch


@pytest.fixture
def sample_data():
    """Create sample training and validation data"""
    np.random.seed(42)
    X_train = np.random.randn(200, 10)
    y_train = np.random.randint(0, 2, 200)
    X_val = np.random.randn(50, 10)
    y_val = np.random.randint(0, 2, 50)

    return X_train, y_train, X_val, y_val


@pytest.fixture
def imbalanced_data():
    """Create imbalanced training data"""
    np.random.seed(42)
    X_train = np.random.randn(200, 15)
    # 5:1 imbalance
    y_train = np.array([0] * 166 + [1] * 34)
    X_val = np.random.randn(50, 15)
    y_val = np.array([0] * 42 + [1] * 8)

    return X_train, y_train, X_val, y_val


class TestImbalanceStrategy:
    """Tests for ImbalanceStrategy enum"""

    def test_strategy_values(self):
        """Test that strategies have correct values"""
        assert mo.ImbalanceStrategy.NONE.value == "none"
        assert mo.ImbalanceStrategy.SMOTE_FULL.value == "smote_full"
        assert mo.ImbalanceStrategy.SMOTE_PARTIAL.value == "smote_partial"
        assert mo.ImbalanceStrategy.CLASS_WEIGHTS.value == "class_weights"
        assert mo.ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS.value == "smote_partial+weights"


class TestImbalanceTrainingResult:
    """Tests for ImbalanceTrainingResult dataclass"""

    def test_result_creation(self):
        """Test creating a training result"""
        result = mo.ImbalanceTrainingResult(
            strategy=mo.ImbalanceStrategy.SMOTE_FULL,
            history=Mock(),
            model=Mock(),
            X_train_final=np.array([[1, 2], [3, 4]]),
            y_train_final=np.array([0, 1]),
            X_train_original=np.array([[1, 2]]),
            y_train_original=np.array([0]),
            class_weight_dict=None,
            smote_ratio=1.0,
            samples_before=1,
            samples_after=2,
            class_dist_before={0: 1, 1: 0},
            class_dist_after={0: 1, 1: 1},
            val_metrics=mo.ValidationMetrics(accuracy=0.75, precision=0.7, recall=0.8, f1=0.74, roc_auc=0.85)
        )

        assert result.strategy == mo.ImbalanceStrategy.SMOTE_FULL
        assert result.samples_after == 2
        assert result.val_metrics.accuracy == 0.75

    def test_result_summary(self):
        """Test result summary generation"""
        result = mo.ImbalanceTrainingResult(
            strategy=mo.ImbalanceStrategy.CLASS_WEIGHTS,
            history=Mock(),
            model=Mock(),
            X_train_final=np.array([[1, 2]]),
            y_train_final=np.array([0]),
            X_train_original=np.array([[1, 2]]),
            y_train_original=np.array([0]),
            class_weight_dict={0: 0.6, 1: 3.0},
            smote_ratio=None,
            samples_before=100,
            samples_after=100,
            class_dist_before={0: 80, 1: 20},
            class_dist_after={0: 80, 1: 20},
            val_metrics={'accuracy': 0.8}
        )

        # Should not raise exception
        result.summary()


class TestTrainWithImbalanceHandling:
    """Tests for train_with_imbalance_handling function"""

    @patch('model_optimizer.SMOTE')
    @patch('model_trainer.train_model_with_class_weights')
    def test_strategy_none(self, mock_train, mock_smote, sample_data):
        """Test 'none' strategy - no imbalance handling"""
        X_train, y_train, X_val, y_val = sample_data
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(50, 1)

        mock_train.return_value = Mock()

        result = mo.train_with_imbalance_handling(
            model=mock_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy='none',
            verbose=False
        )

        # Should not use SMOTE
        mock_smote.assert_not_called()

        # Should train without class weights
        assert mock_train.called
        call_args = mock_train.call_args
        assert call_args[0][5] is None  # class_weights should be None (6th positional arg)

        # Check result
        assert result.strategy == mo.ImbalanceStrategy.NONE
        assert result.samples_after == result.samples_before

    @patch('model_optimizer.SMOTE')
    @patch('model_trainer.train_model_with_class_weights')
    def test_strategy_smote_full(self, mock_train, mock_smote, imbalanced_data):
        """Test 'smote_full' strategy"""
        X_train, y_train, X_val, y_val = imbalanced_data
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(50, 1)

        # Mock SMOTE
        mock_smote_instance = Mock()
        mock_smote_instance.fit_resample.return_value = (
            np.random.randn(332, 15),  # Balanced data
            np.array([0] * 166 + [1] * 166)
        )
        mock_smote.return_value = mock_smote_instance
        mock_train.return_value = Mock()

        result = mo.train_with_imbalance_handling(
            model=mock_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy='smote_full',
            verbose=False
        )

        # Should use SMOTE with ratio=1.0
        mock_smote.assert_called_once()
        assert mock_smote.call_args[1]['sampling_strategy'] == 1.0

        # Result should show increased samples
        assert result.samples_after > result.samples_before
        assert result.smote_ratio == 1.0

    @patch('model_optimizer.compute_class_weight')
    @patch('model_trainer.train_model_with_class_weights')
    def test_strategy_class_weights(self, mock_train, mock_compute_weights, sample_data):
        """Test 'class_weights' strategy"""
        X_train, y_train, X_val, y_val = sample_data
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(50, 1)

        mock_compute_weights.return_value = np.array([0.6, 3.0])
        mock_train.return_value = Mock()

        result = mo.train_with_imbalance_handling(
            model=mock_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy='class_weights',
            auto_calculate_weights=True,
            verbose=False
        )

        # Should calculate class weights
        assert mock_compute_weights.called

        # Should pass weights to training
        call_args = mock_train.call_args
        assert call_args[0][5] is not None  # class_weights should not be None (6th positional arg)

        # Check result
        assert result.strategy == mo.ImbalanceStrategy.CLASS_WEIGHTS
        assert result.class_weight_dict is not None

    @patch('model_optimizer.SMOTE')
    @patch('model_optimizer.compute_class_weight')
    @patch('model_trainer.train_model_with_class_weights')
    def test_strategy_combined(self, mock_train, mock_compute_weights, mock_smote, imbalanced_data):
        """Test 'smote_partial+weights' strategy"""
        X_train, y_train, X_val, y_val = imbalanced_data
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(50, 1)

        # Mock SMOTE
        mock_smote_instance = Mock()
        mock_smote_instance.fit_resample.return_value = (
            np.random.randn(249, 15),
            np.array([0] * 166 + [1] * 83)  # Partial balance
        )
        mock_smote.return_value = mock_smote_instance

        mock_compute_weights.return_value = np.array([0.6, 3.0])
        mock_train.return_value = Mock()

        result = mo.train_with_imbalance_handling(
            model=mock_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy='smote_partial+weights',
            smote_ratio=0.5,
            auto_calculate_weights=True,
            verbose=False
        )

        # Should use both SMOTE and class weights
        assert mock_smote.called
        assert mock_compute_weights.called

        # Check result
        assert result.strategy == mo.ImbalanceStrategy.SMOTE_PARTIAL_WEIGHTS
        assert result.class_weight_dict is not None
        assert result.smote_ratio == 0.5

    def test_invalid_strategy(self, sample_data):
        """Test that invalid strategy raises error"""
        X_train, y_train, X_val, y_val = sample_data
        mock_model = Mock()

        with pytest.raises(ValueError, match="Unknown strategy"):
            mo.train_with_imbalance_handling(
                model=mock_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                strategy='invalid_strategy'
            )

    @patch('model_trainer.train_model_with_class_weights')
    def test_validation_metrics_calculated(self, mock_train, sample_data):
        """Test that validation metrics are calculated"""
        X_train, y_train, X_val, y_val = sample_data
        mock_model = Mock()
        # Mock predictions
        mock_model.predict.return_value = np.random.rand(50, 1)
        mock_train.return_value = Mock()

        result = mo.train_with_imbalance_handling(
            model=mock_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy='none',
            verbose=False
        )

        # Check that metrics were calculated
        assert 'accuracy' in result.val_metrics
        assert 'precision' in result.val_metrics
        assert 'recall' in result.val_metrics
        assert 'f1' in result.val_metrics
        assert 'roc_auc' in result.val_metrics


class TestOptimizationComparison:
    """Tests for OptimizationComparison dataclass"""

    def test_comparison_creation(self):
        """Test creating optimization comparison"""
        mock_result1 = Mock()
        mock_result1.val_metrics = mo.ValidationMetrics(accuracy=0.7, precision=0.72, recall=0.73, f1=0.75, roc_auc=0.78)

        mock_result2 = Mock()
        mock_result2.val_metrics = mo.ValidationMetrics(accuracy=0.78, precision=0.79, recall=0.81, f1=0.80, roc_auc=0.82)

        results = {
            'strategy1': mock_result1,
            'strategy2': mock_result2
        }

        df = pd.DataFrame([
            {'strategy': 'strategy2', 'f1': 0.80},
            {'strategy': 'strategy1', 'f1': 0.75}
        ])

        comparison = mo.OptimizationComparison(
            results=results,
            comparison_df=df,
            best_strategy='strategy2',
            best_metric='f1',
            ranking=[('strategy2', 0.80), ('strategy1', 0.75)]
        )

        assert comparison.best_strategy == 'strategy2'
        assert len(comparison.results) == 2

    def test_comparison_print(self):
        """Test comparison printing"""
        mock_result = Mock()
        mock_result.val_metrics = mo.ValidationMetrics(accuracy=0.7, precision=0.72, recall=0.73, f1=0.75, roc_auc=0.78)
        mock_result.model = Mock()

        results = {'strategy1': mock_result}
        df = pd.DataFrame([{'strategy': 'strategy1', 'f1': 0.75}])

        comparison = mo.OptimizationComparison(
            results=results,
            comparison_df=df,
            best_strategy='strategy1',
            best_metric='f1',
            ranking=[('strategy1', 0.75)]
        )

        # Should not raise exception
        comparison.print_comparison()

    def test_get_best_model(self):
        """Test getting best model from comparison"""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.model = mock_model

        results = {'best': mock_result}
        df = pd.DataFrame([{'strategy': 'best', 'f1': 0.8}])

        comparison = mo.OptimizationComparison(
            results=results,
            comparison_df=df,
            best_strategy='best',
            best_metric='f1',
            ranking=[('best', 0.8)]
        )

        best_model = comparison.get_best_model()
        assert best_model == mock_model


class TestOptimizeImbalanceStrategy:
    """Tests for optimize_imbalance_strategy function"""

    def test_optimize_multiple_strategies(self, imbalanced_data):
        """Test optimizing across multiple strategies"""
        X_train, y_train, X_val, y_val = imbalanced_data

        def build_model():
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=5, max_depth=3)

        def train_sklearn_model(model, X_train, y_train, X_val, y_val, class_weights, epochs, callbacks):
            """Custom training function for sklearn models"""
            # sklearn uses sample_weight, not class_weight dict
            sample_weight = None
            if class_weights is not None:
                sample_weight = [class_weights[y] for y in y_train]
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return None  # No history object for sklearn

        comparison = mo.optimize_imbalance_strategy(
            model_builder=build_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategies=['none', 'class_weights'],
            optimize_for='f1',
            epochs=2,
            train_fn=train_sklearn_model,
            verbose=False
        )

        assert comparison is not None
        assert len(comparison.results) == 2
        assert comparison.best_strategy in ['none', 'class_weights']

    def test_optimize_for_different_metrics(self, sample_data):
        """Test optimizing for different metrics"""
        X_train, y_train, X_val, y_val = sample_data

        def build_model():
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression()

        def train_sklearn_model(model, X_train, y_train, X_val, y_val, class_weights, epochs, callbacks):
            """Custom training function for sklearn models"""
            sample_weight = None
            if class_weights is not None:
                sample_weight = [class_weights[y] for y in y_train]
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return None

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            comparison = mo.optimize_imbalance_strategy(
                model_builder=build_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                strategies=['none'],
                train_fn=train_sklearn_model,
                optimize_for=metric,
                verbose=False
            )

            assert comparison.best_metric == metric


class TestIntegration:
    """Integration tests with minimal dependencies"""

    @patch('model_trainer.train_model_with_class_weights')
    def test_full_pipeline_with_mocks(self, mock_train, imbalanced_data):
        """Test full pipeline with mocked training"""
        X_train, y_train, X_val, y_val = imbalanced_data

        # Mock model and training
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(50, 1)
        mock_train.return_value = Mock()

        # Test single strategy
        result = mo.train_with_imbalance_handling(
            model=mock_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy='class_weights',
            auto_calculate_weights=True,
            verbose=False
        )

        # Validate complete result
        assert result.strategy == mo.ImbalanceStrategy.CLASS_WEIGHTS
        assert result.samples_before == len(X_train)
        assert result.class_dist_before is not None
        assert result.class_dist_after is not None
        assert len(result.val_metrics) == 5  # accuracy, precision, recall, f1, roc_auc

        # Test summary doesn't crash
        result.summary()

    def test_class_distribution_tracking(self, imbalanced_data):
        """Test that class distributions are tracked correctly"""
        X_train, y_train, X_val, y_val = imbalanced_data

        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(50, 1)

        with patch('model_trainer.train_model_with_class_weights'):
            result = mo.train_with_imbalance_handling(
                model=mock_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                strategy='none',
                verbose=False
            )

            # Check original distribution
            assert 0 in result.class_dist_before
            assert 1 in result.class_dist_before
            assert result.class_dist_before[0] == 166
            assert result.class_dist_before[1] == 34

            # For 'none' strategy, before and after should be same
            assert result.class_dist_before == result.class_dist_after
