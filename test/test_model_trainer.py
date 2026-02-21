"""
Pytest tests for model_trainer module
"""

import pytest
import numpy as np
import model_trainer as mt


@pytest.fixture
def sample_training_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    X_val = np.random.randn(30, n_features)
    y_val = np.random.randint(0, 2, 30)

    return X_train, y_train, X_val, y_val


@pytest.fixture
def imbalanced_data():
    """Create imbalanced training data"""
    np.random.seed(42)
    X_train = np.random.randn(200, 15)
    y_train = np.array([0] * 160 + [1] * 40)  # 4:1 imbalance
    X_val = np.random.randn(50, 15)
    y_val = np.array([0] * 40 + [1] * 10)

    return X_train, y_train, X_val, y_val


class TestClassWeights:
    """Tests for class weight calculations"""

    def test_calculate_class_weights_balanced(self):
        """Test class weight calculation with balanced data"""
        y_train = np.array([0] * 50 + [1] * 50)

        class_weight_dict, class_weights = mt.calculate_class_weights(y_train)

        # For balanced data, weights should be close to 1
        assert abs(class_weights[0] - 1.0) < 0.1
        assert abs(class_weights[1] - 1.0) < 0.1

        # Check dictionary
        assert class_weight_dict[0] == class_weights[0]
        assert class_weight_dict[1] == class_weights[1]

    def test_calculate_class_weights_imbalanced(self):
        """Test class weight calculation with imbalanced data"""
        # 5:1 imbalance ratio
        y_train = np.array([0] * 500 + [1] * 100)

        class_weight_dict, class_weights = mt.calculate_class_weights(y_train)

        # Minority class should have higher weight
        assert class_weights[1] > class_weights[0]

        # Weight ratio should be approximately equal to class imbalance ratio
        weight_ratio = class_weights[1] / class_weights[0]
        assert abs(weight_ratio - 5.0) < 0.5

    def test_calculate_class_weights_returns_dict(self):
        """Test that function returns proper dictionary"""
        y_train = np.array([0, 1, 0, 1, 0, 1])

        class_weight_dict, class_weights = mt.calculate_class_weights(y_train)

        assert isinstance(class_weight_dict, dict)
        assert 0 in class_weight_dict
        assert 1 in class_weight_dict


@pytest.mark.skipif(
    not hasattr(mt, 'build_neural_network'),
    reason="TensorFlow not available"
)
class TestNeuralNetworkBuilder:
    """Tests for neural network building (requires TensorFlow)"""

    def test_build_neural_network_default_params(self):
        """Test building network with default parameters"""
        try:
            model = mt.build_neural_network(input_dim=17)

            assert model is not None
            assert len(model.layers) == 5  # 2 dense + 2 dropout + 1 output

            # Check input shape
            assert model.input_shape[1] == 17

            # Check output shape
            assert model.output_shape[1] == 1
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_build_neural_network_custom_layers(self):
        """Test building network with custom layer sizes"""
        try:
            model = mt.build_neural_network(
                input_dim=20,
                layers=[64, 32, 16],
                dropout_rate=0.4,
                learning_rate=0.0001
            )

            assert model is not None
            # 3 dense hidden + 3 dropout + 1 output = 7 layers
            assert len(model.layers) == 7
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_build_neural_network_single_layer(self):
        """Test building network with single hidden layer"""
        try:
            model = mt.build_neural_network(
                input_dim=10,
                layers=[32]
            )

            assert model is not None
            # 1 dense hidden + 1 dropout + 1 output = 3 layers
            assert len(model.layers) == 3
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_build_neural_network_is_compiled(self):
        """Test that built model is compiled"""
        try:
            model = mt.build_neural_network(input_dim=17)

            # Check that model has optimizer (sign of compilation)
            assert model.optimizer is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_build_neural_network_has_correct_activation(self):
        """Test that output layer has sigmoid activation"""
        try:
            model = mt.build_neural_network(input_dim=17)

            # Get last layer
            output_layer = model.layers[-1]

            # Check activation is sigmoid
            assert output_layer.activation.__name__ == 'sigmoid'
        except ImportError:
            pytest.skip("TensorFlow not installed")


@pytest.mark.skipif(
    not hasattr(mt, 'create_early_stopping'),
    reason="TensorFlow not available"
)
class TestEarlyStopping:
    """Tests for early stopping callback (requires TensorFlow)"""

    def test_create_early_stopping(self):
        """Test creating early stopping callback"""
        try:
            callback = mt.create_early_stopping(
                patience=5,
                monitor='val_auc',
                mode='max'
            )

            assert callback is not None
            assert callback.patience == 5
            assert callback.monitor == 'val_auc'
            assert callback.mode == 'max'
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_create_early_stopping_default_params(self):
        """Test early stopping with default parameters"""
        try:
            callback = mt.create_early_stopping()

            assert callback.patience == 5
            assert callback.monitor == 'val_auc'
            assert callback.restore_best_weights
        except ImportError:
            pytest.skip("TensorFlow not installed")


@pytest.mark.skipif(
    not hasattr(mt, 'train_model_with_class_weights'),
    reason="TensorFlow not available"
)
class TestModelTraining:
    """Tests for model training functions (requires TensorFlow)"""

    def test_train_model_with_class_weights(self, sample_training_data):
        """Test model training with class weights"""
        try:
            X_train, y_train, X_val, y_val = sample_training_data
            class_weights = {0: 0.6, 1: 3.0}

            model = mt.build_neural_network(
                input_dim=X_train.shape[1],
                layers=[8, 4]
            )

            history = mt.train_model_with_class_weights(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                class_weights,
                epochs=2,
                batch_size=32,
                verbose=0
            )

            assert history is not None
            assert 'loss' in history.history
            assert 'val_loss' in history.history
            assert len(history.history['loss']) == 2  # 2 epochs
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_train_model_with_callbacks(self, sample_training_data):
        """Test model training with custom callbacks"""
        try:
            X_train, y_train, X_val, y_val = sample_training_data
            class_weights = {0: 0.6, 1: 3.0}

            model = mt.build_neural_network(input_dim=X_train.shape[1])
            early_stop = mt.create_early_stopping(patience=1, verbose=0)

            history = mt.train_model_with_class_weights(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                class_weights,
                epochs=10,
                callbacks=[early_stop],
                verbose=0
            )

            # With early stopping, should stop before 10 epochs
            assert len(history.history['loss']) <= 10
        except ImportError:
            pytest.skip("TensorFlow not installed")


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_get_model_config_name(self):
        """Test model configuration name generation"""
        if not hasattr(mt, 'get_model_config_name'):
            pytest.skip("get_model_config_name not available")

        name = mt.get_model_config_name(
            layers=[32, 16],
            dropout=0.3,
            lr=0.001
        )

        assert isinstance(name, str)
        assert '32' in name
        assert '16' in name
        assert '0.3' in name
        assert '0.001' in name

    def test_get_model_config_name_different_configs(self):
        """Test that different configs produce different names"""
        if not hasattr(mt, 'get_model_config_name'):
            pytest.skip("get_model_config_name not available")

        name1 = mt.get_model_config_name([32, 16], 0.3, 0.001)
        name2 = mt.get_model_config_name([64, 32], 0.3, 0.001)
        name3 = mt.get_model_config_name([32, 16], 0.4, 0.001)

        assert name1 != name2
        assert name1 != name3

    def test_calculate_class_weights_returns_both_formats(self):
        """Test that function returns both dict and array"""
        y_train = np.array([0] * 70 + [1] * 30)

        class_weight_dict, class_weights = mt.calculate_class_weights(y_train)

        # Check dictionary format
        assert isinstance(class_weight_dict, dict)
        assert len(class_weight_dict) == 2

        # Check array format
        assert isinstance(class_weights, np.ndarray)
        assert len(class_weights) == 2

    def test_class_weights_handle_extreme_imbalance(self):
        """Test class weights with extreme imbalance"""
        y_train = np.array([0] * 990 + [1] * 10)  # 99:1 imbalance

        class_weight_dict, class_weights = mt.calculate_class_weights(y_train)

        # Minority class should have much higher weight
        assert class_weights[1] > class_weights[0] * 50


@pytest.mark.skipif(
    not hasattr(mt, 'build_neural_network'),
    reason="TensorFlow not available"
)
class TestIntegration:
    """Integration tests (requires TensorFlow)"""

    def test_full_training_pipeline(self, imbalanced_data):
        """Test complete training pipeline"""
        try:
            X_train, y_train, X_val, y_val = imbalanced_data

            # Calculate class weights
            class_weight_dict, _ = mt.calculate_class_weights(y_train)

            # Build model
            model = mt.build_neural_network(
                input_dim=X_train.shape[1],
                layers=[16, 8],
                dropout_rate=0.3,
                learning_rate=0.001
            )

            # Train model
            early_stop = mt.create_early_stopping(patience=2, verbose=0)

            history = mt.train_model_with_class_weights(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                class_weight_dict,
                epochs=5,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            # Validate results
            assert history is not None
            assert 'loss' in history.history
            assert 'accuracy' in history.history

            # Model should be trained (loss should be > 0)
            assert history.history['loss'][0] > 0
        except ImportError:
            pytest.skip("TensorFlow not installed")
