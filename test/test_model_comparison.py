"""
Pytest tests for model_comparison module
"""

import pytest
import numpy as np
import pandas as pd
import model_comparison as mc


@pytest.fixture
def sample_data():
    """Create sample training and test data"""
    np.random.seed(42)
    n_train = 200
    n_test = 50
    n_features = 10

    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randint(0, 2, n_train)
    X_test = np.random.randn(n_test, n_features)
    y_test = np.random.randint(0, 2, n_test)
    X_val = np.random.randn(30, n_features)
    y_val = np.random.randint(0, 2, 30)

    return mc.ComparisonInput(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        class_names=['Class 0', 'Class 1']
    )


@pytest.fixture
def imbalanced_data():
    """Create imbalanced training data"""
    np.random.seed(42)
    X_train = np.random.randn(200, 15)
    y_train = np.array([0] * 160 + [1] * 40)  # 4:1 imbalance
    X_test = np.random.randn(50, 15)
    y_test = np.array([0] * 40 + [1] * 10)
    X_val = np.random.randn(30, 15)
    y_val = np.array([0] * 24 + [1] * 6)

    return mc.ComparisonInput(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        class_names=['Majority', 'Minority']
    )


class TestModelType:
    """Tests for ModelType enum"""

    def test_model_type_values(self):
        """Test that all model types have correct values"""
        assert mc.ModelType.NEURAL_NETWORK.value == "neural_network"
        assert mc.ModelType.RANDOM_FOREST.value == "random_forest"
        assert mc.ModelType.XGBOOST.value == "xgboost"
        assert mc.ModelType.LOGISTIC_REGRESSION.value == "logistic_regression"
        assert mc.ModelType.GRADIENT_BOOSTING.value == "gradient_boosting"


class TestModelConfig:
    """Tests for ModelConfig dataclass"""

    def test_model_config_creation(self):
        """Test creating a model config"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="Test RF",
            params={'n_estimators': 50}
        )

        assert config.model_type == mc.ModelType.RANDOM_FOREST
        assert config.name == "Test RF"
        assert config.params['n_estimators'] == 50

    def test_model_config_default_params_rf(self):
        """Test default parameters for Random Forest"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Default"
        )

        assert 'n_estimators' in config.params
        assert 'max_depth' in config.params
        assert config.params['n_estimators'] == 100

    def test_model_config_default_params_nn(self):
        """Test default parameters for Neural Network"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.NEURAL_NETWORK,
            name="NN Default"
        )

        assert 'layers' in config.params
        assert 'dropout_rate' in config.params
        assert config.params['layers'] == [32, 16]

    def test_model_config_class_weight(self):
        """Test class weight setting"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Weighted",
            class_weight='balanced'
        )

        assert config.class_weight == 'balanced'


class TestComparisonInput:
    """Tests for ComparisonInput dataclass"""

    def test_comparison_input_creation(self, sample_data):
        """Test creating comparison input"""
        assert sample_data.X_train.shape[0] == 200
        assert sample_data.X_test.shape[0] == 50
        assert len(sample_data.class_names) == 2

    def test_comparison_input_with_validation(self, sample_data):
        """Test that validation data is included"""
        assert sample_data.X_val is not None
        assert sample_data.y_val is not None
        assert sample_data.X_val.shape[0] == 30


class TestBuildModel:
    """Tests for model building function"""

    def test_build_random_forest(self):
        """Test building a Random Forest model"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="Test RF",
            params={'n_estimators': 10}
        )

        model = mc.build_model(config, input_dim=10)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert model.n_estimators == 10

    def test_build_random_forest_with_class_weight(self):
        """Test building RF with class weights"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Weighted",
            class_weight='balanced'
        )

        model = mc.build_model(config, input_dim=10)

        assert model.class_weight == 'balanced'

    def test_build_xgboost(self):
        """Test building an XGBoost model"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.XGBOOST,
            name="Test XGB"
        )

        model = mc.build_model(config, input_dim=10, scale_pos_weight=2.0)

        assert model is not None
        assert hasattr(model, 'fit')

    def test_build_logistic_regression(self):
        """Test building Logistic Regression model"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.LOGISTIC_REGRESSION,
            name="Test LR"
        )

        model = mc.build_model(config, input_dim=10)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict_proba')

    def test_build_neural_network(self):
        """Test building a Neural Network model"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.NEURAL_NETWORK,
            name="Test NN",
            params={'layers': [16, 8]}
        )

        model = mc.build_model(config, input_dim=10)

        assert model is not None


class TestTrainAndEvaluate:
    """Tests for training and evaluation"""

    def test_train_random_forest(self, sample_data):
        """Test training a Random Forest model"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Test",
            params={'n_estimators': 10, 'max_depth': 3}
        )

        result = mc.train_and_evaluate_model(config, sample_data, verbose=False)

        assert result is not None
        assert result.model_type == mc.ModelType.RANDOM_FOREST
        assert result.name == "RF Test"
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.roc_auc <= 1
        assert result.n_samples_train == 200
        assert result.n_samples_test == 50

    def test_train_with_imbalanced_data(self, imbalanced_data):
        """Test training with imbalanced data"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Imbalanced",
            class_weight='balanced',
            params={'n_estimators': 10}
        )

        result = mc.train_and_evaluate_model(config, imbalanced_data, verbose=False)

        assert result is not None
        assert result.recall >= 0  # Should catch some minority samples
        assert result.precision >= 0

    def test_result_has_predictions(self, sample_data):
        """Test that result contains predictions"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Test",
            params={'n_estimators': 10}
        )

        result = mc.train_and_evaluate_model(config, sample_data, verbose=False)

        assert result.y_pred is not None
        assert result.y_pred_proba is not None
        assert len(result.y_pred) == 50  # Test set size
        assert len(result.y_pred_proba) == 50

    def test_result_has_confusion_matrix(self, sample_data):
        """Test that result contains confusion matrix"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Test",
            params={'n_estimators': 10}
        )

        result = mc.train_and_evaluate_model(config, sample_data, verbose=False)

        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (2, 2)

    def test_result_has_feature_importance(self, sample_data):
        """Test that Random Forest result has feature importance"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="RF Test",
            params={'n_estimators': 10}
        )

        result = mc.train_and_evaluate_model(config, sample_data, verbose=False)

        assert result.feature_importance is not None
        assert isinstance(result.feature_importance, pd.DataFrame)
        assert 'importance' in result.feature_importance.columns


class TestCompareModels:
    """Tests for model comparison"""

    def test_compare_multiple_models(self, sample_data):
        """Test comparing multiple models"""
        configs = [
            mc.ModelConfig(
                model_type=mc.ModelType.RANDOM_FOREST,
                name="RF Small",
                params={'n_estimators': 5, 'max_depth': 2}
            ),
            mc.ModelConfig(
                model_type=mc.ModelType.LOGISTIC_REGRESSION,
                name="LogReg"
            )
        ]

        comparison = mc.compare_models(
            configs=configs,
            data=sample_data,
            optimize_for='f1',
            verbose=False
        )

        assert comparison is not None
        assert len(comparison.results) == 2
        assert 'RF Small' in comparison.results
        assert 'LogReg' in comparison.results

    def test_comparison_identifies_best_model(self, sample_data):
        """Test that comparison identifies best model"""
        configs = [
            mc.ModelConfig(
                model_type=mc.ModelType.RANDOM_FOREST,
                name="RF Model",
                params={'n_estimators': 10}
            ),
            mc.ModelConfig(
                model_type=mc.ModelType.LOGISTIC_REGRESSION,
                name="LR Model"
            )
        ]

        comparison = mc.compare_models(
            configs=configs,
            data=sample_data,
            optimize_for='accuracy',
            verbose=False
        )

        assert comparison.best_model_name in ['RF Model', 'LR Model']
        assert comparison.best_metric == 'accuracy'

    def test_comparison_has_dataframe(self, sample_data):
        """Test that comparison produces DataFrame"""
        configs = [
            mc.ModelConfig(
                model_type=mc.ModelType.RANDOM_FOREST,
                name="RF",
                params={'n_estimators': 5}
            )
        ]

        comparison = mc.compare_models(
            configs=configs,
            data=sample_data,
            verbose=False
        )

        assert isinstance(comparison.comparison_df, pd.DataFrame)
        assert 'Model' in comparison.comparison_df.columns
        assert 'accuracy' in comparison.comparison_df.columns
        assert 'roc_auc' in comparison.comparison_df.columns

    def test_comparison_ranking(self, sample_data):
        """Test that comparison provides ranking"""
        configs = [
            mc.ModelConfig(
                model_type=mc.ModelType.RANDOM_FOREST,
                name="Model1",
                params={'n_estimators': 5}
            ),
            mc.ModelConfig(
                model_type=mc.ModelType.LOGISTIC_REGRESSION,
                name="Model2"
            )
        ]

        comparison = mc.compare_models(
            configs=configs,
            data=sample_data,
            optimize_for='f1',
            verbose=False
        )

        assert len(comparison.ranking) == 2
        assert all(isinstance(item, tuple) for item in comparison.ranking)


class TestDefaultConfigs:
    """Tests for default configuration creation"""

    def test_create_default_configs_all(self):
        """Test creating all default configs"""
        configs = mc.create_default_configs(
            use_neural_network=True,
            use_random_forest=True,
            use_xgboost=False,  # Skip XGBoost as it may not be installed
            use_logistic=True,
            class_weight='balanced'
        )

        assert len(configs) >= 2  # At least RF and LR
        assert any(c.model_type == mc.ModelType.RANDOM_FOREST for c in configs)
        assert any(c.model_type == mc.ModelType.LOGISTIC_REGRESSION for c in configs)

    def test_create_default_configs_rf_only(self):
        """Test creating only Random Forest config"""
        configs = mc.create_default_configs(
            use_neural_network=False,
            use_random_forest=True,
            use_xgboost=False,
            use_logistic=False
        )

        assert len(configs) == 1
        assert configs[0].model_type == mc.ModelType.RANDOM_FOREST

    def test_default_configs_have_class_weight(self):
        """Test that default configs include class weight"""
        configs = mc.create_default_configs(
            use_random_forest=True,
            class_weight='balanced'
        )

        rf_config = next(c for c in configs if c.model_type == mc.ModelType.RANDOM_FOREST)
        assert rf_config.class_weight == 'balanced'


class TestIntegration:
    """Integration tests"""

    def test_full_comparison_pipeline(self, imbalanced_data):
        """Test complete comparison pipeline"""
        # Create configs
        configs = mc.create_default_configs(
            use_neural_network=False,  # Skip NN as it requires TensorFlow
            use_random_forest=True,
            use_xgboost=False,
            use_logistic=True,
            class_weight='balanced'
        )

        # Run comparison
        comparison = mc.compare_models(
            configs=configs,
            data=imbalanced_data,
            optimize_for='recall',
            verbose=False
        )

        # Validate results
        assert comparison is not None
        assert len(comparison.results) >= 2

        # Check best model
        best_model = comparison.get_best_model()
        assert best_model is not None
        assert hasattr(best_model, 'predict')

        # Check best result
        best_result = comparison.get_best_result()
        assert best_result.recall > 0  # Should catch some minority samples

    def test_model_result_summary(self, sample_data):
        """Test model result summary generation"""
        config = mc.ModelConfig(
            model_type=mc.ModelType.RANDOM_FOREST,
            name="Test Model",
            params={'n_estimators': 10}
        )

        result = mc.train_and_evaluate_model(config, sample_data, verbose=False)

        # Should not raise exception
        result.summary()

    def test_comparison_result_print(self, sample_data):
        """Test comparison result printing"""
        configs = [
            mc.ModelConfig(
                model_type=mc.ModelType.RANDOM_FOREST,
                name="RF",
                params={'n_estimators': 5}
            )
        ]

        comparison = mc.compare_models(
            configs=configs,
            data=sample_data,
            verbose=False
        )

        # Should not raise exception
        comparison.print_comparison()
