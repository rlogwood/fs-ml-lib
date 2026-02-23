"""
Model Training Module for Deep Learning Models

This module provides functions for building, training, and configuring
neural network models for loan default prediction.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

try:
    from . import text_util as tu
except ImportError:
    import text_util as tu

# Define F1 metric class (Keras doesn't have built-in F1)
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def calculate_class_weights(y_train, class_names=None):
    """
    Calculate balanced class weights for imbalanced datasets.

    Parameters:
    -----------
    y_train : array-like
        Training labels
    class_names : dict, optional
        Dictionary mapping class labels to names (e.g., {0: 'Paid', 1: 'Default'})
        If None, defaults to {0: 'Paid', 1: 'Default'}

    Returns:
    --------
    dict
        Dictionary mapping class labels to weights
    numpy.ndarray
        Array of class weights

    Examples:
    ---------
    >>> class_weight_dict, class_weights = calculate_class_weights(y_train)
    >>> print(f"Class 0 weight: {class_weight_dict[0]:.4f}")

    Insight:
    --------
    2 classes is typical because:
    - Most classification problems are binary (yes/no, fraud/legitimate, default/paid, sick/healthy)

    The special handling for 2 classes provides a more intuitive interpretation with
    the ratio and the sentence explaining penalty differences

    More than 2 classes occurs in multi-class classification:
    - Credit ratings: AAA, AA, A, BBB, BB, B, CCC (7 classes)
    - Loan risk tiers: Low, Medium, High, Very High (4 classes)
    - Customer segments: Bronze, Silver, Gold, Platinum (4 classes)
    - Image classification: cat, dog, bird, fish, etc.
    - Sentiment analysis: Very Negative, Negative, Neutral, Positive, Very Positive (5 classes)
    - Medical diagnosis: Multiple disease types or severity levels

    Why the code handles both: The compute_class_weight function from sklearn works
    with any number of classes, so the function naturally supports multi-class scenarios.
    The special 2-class formatting just makes the output clearer for the most common case -
    showing a simple ratio and readable interpretation instead of just listing all weights.
    In your loan default prediction case, you're doing binary classification (paid vs default),
    so you'd typically stay with 2 classes. But if you later wanted to predict something like
    paid on time / paid late / defaulted" (3 classes), the function would handle it automatically.
    """

    tu.print_heading("CLASS WEIGHT CALCULATION")

    # Set default class names if not provided
    if class_names is None:
        class_names = {0: 'Paid', 1: 'Default'}

    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    # Create dictionary
    class_weight_dict = dict(zip(classes, class_weights))

    print("\nCalculated Class Weights:")
    for cls, weight in class_weight_dict.items():
        class_name = class_names.get(cls, f'Class {cls}')
        print(f"  Class {cls} ({class_name}): {weight:.4f}")

    # Calculate and display weight ratios for all pairs (if more than 2 classes)
    if len(classes) == 2:
        ratio = class_weights[1] / class_weights[0]
        name_0 = class_names.get(classes[0], f'Class {classes[0]}')
        name_1 = class_names.get(classes[1], f'Class {classes[1]}')
        print(f"\n  Weight Ratio ({classes[1]}/{classes[0]}): {ratio:.2f}x")
        print("\n  Interpretation: The model will penalize misclassifying")
        print(f"  a {name_1.lower()} {ratio:.2f}x more than misclassifying a {name_0.lower()}.")
    else:
        print("\n  Weight Ratios:")
        for i, cls in enumerate(classes):
            name = class_names.get(cls, f'Class {cls}')
            print(f"    {name}: {class_weights[i]:.4f}")

    return class_weight_dict, class_weights

def build_neural_network(input_dim, layers=[32, 16], dropout_rate=0.3, learning_rate=0.001):
    """
    Build a sequential neural network for binary classification.

    Parameters:
    -----------
    input_dim : int
        Number of input features
    layers : list
        List of integers specifying units in each hidden layer
    dropout_rate : float
        Dropout rate for regularization (0.0 to 1.0)
    learning_rate : float
        Learning rate for optimizer

    Returns:
    --------
    keras.Model
        Compiled Keras model

    Examples:
    ---------
    >>> model = build_neural_network(input_dim=17, layers=[64, 32], dropout_rate=0.3)
    >>> model.summary()
    """
    from tensorflow import keras
    from tensorflow.keras import layers as keras_layers

    tu.print_heading("BUILDING NEURAL NETWORK MODEL")

    # Clear any previous models
    keras.backend.clear_session()

    # Create sequential model
    model = keras.Sequential(name='sequential')

    # Add input layer and first hidden layer
    model.add(keras_layers.Dense(
        layers[0],
        activation='relu',
        input_shape=(input_dim,),
        name='hidden_layer_1'
    ))
    model.add(keras_layers.Dropout(dropout_rate, name='dropout_1'))

    # Add additional hidden layers
    for i, units in enumerate(layers[1:], start=2):
        model.add(keras_layers.Dense(
            units,
            activation='relu',
            name=f'hidden_layer_{i}'
        ))
        model.add(keras_layers.Dropout(dropout_rate, name=f'dropout_{i}'))

    # Output layer
    model.add(keras_layers.Dense(1, activation='sigmoid', name='output'))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            F1Score(name='f1')
        ]
    )

    print("\nModel Architecture:\n")
    model.summary()

    print(tu.bold_and_colored_text("\nMODEL READY FOR TRAINING", tu.Color.GREEN))

    return model

def create_early_stopping(patience=5, monitor='val_f1', mode='max', verbose=1, restore_best_weights=True):
    """
    Create early stopping callback for training.

    Parameters:
    -----------
    patience : int
        Number of epochs with no improvement after which training will be stopped
    monitor : str
        Metric to monitor
    mode : str
        'min', 'max', or 'auto'
    verbose : int
        Verbosity mode

    Returns:
    --------
    keras.callbacks.EarlyStopping
        Early stopping callback

    Examples:
    ---------
    >>> early_stop = create_early_stopping(patience=5, monitor='val_f1')
    >>> history = model.fit(X, y, callbacks=[early_stop])
    """
    from tensorflow.keras.callbacks import EarlyStopping

    # Validate monitor parameter against known metrics
    valid_monitors = [
        'loss', 'val_loss',
        'accuracy', 'val_accuracy',
        'auc', 'val_auc',
        'precision', 'val_precision',
        'recall', 'val_recall',
        'f1', 'val_f1'
    ]

    if monitor not in valid_monitors:
        raise ValueError(
            f"Invalid monitor: '{monitor}'. "
            f"Must be one of: {', '.join(valid_monitors)}"
        )

    print(tu.bold_and_colored_text(f"\n** CREATING EARLY STOPPING CALLBACK (patience={patience}, monitor='{monitor}', mode='{mode}', verbose={verbose}, restore_best_weights={restore_best_weights})\n", tu.Color.GREEN))

    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights,
        mode=mode,
        verbose=verbose
    )

def train_model_with_class_weights(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    epochs=50,
    batch_size=256,
    callbacks=None,
    model_verbosity=1
):
    """
    Train a neural network model with class weights.

    Parameters:
    -----------
    model : keras.Model
        Compiled Keras model
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    class_weights : dict
        Dictionary mapping class labels to weights
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size for training
    callbacks : list
        List of Keras callbacks
    verbose : int
        Verbosity mode

    Returns:
    --------
    keras.callbacks.History
        Training history object

    Examples:
    ---------
    >>> history = train_model_with_class_weights(
    ...     model, X_train, y_train, X_val, y_val,
    ...     class_weights={0: 0.59, 1: 3.12},
    ...     epochs=50
    ... )
    """
    tu.print_heading("TRAINING MODEL")

    print(f"\nTraining with class weights: {class_weights}")
    print("This will force the model to learn from minority class...\n")

    if callbacks is None:
        callbacks = []

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=model_verbosity
    )

    # Add batch_size to history.params for later extraction by get_model_architecture_info()
    # Keras doesn't store this by default
    params_dict = getattr(history, 'params', None)
    if params_dict and isinstance(params_dict, dict):
        params_dict['batch_size'] = batch_size

    return history


def get_model_config_name(layers, dropout, lr):
    """
    Generate a descriptive name for a model configuration.

    Parameters:
    -----------
    layers : list
        List of layer sizes
    dropout : float
        Dropout rate
    lr : float
        Learning rate

    Returns:
    --------
    str
        Configuration name
    """
    layer_str = f"{layers[0]}-{layers[1]}"
    return f"NN_{layer_str}_d{dropout}_lr{lr}"


if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("=" * 50)
    print("\nThis module provides neural network training functions.")
    print("\nAvailable functions:")
    print("  - calculate_class_weights(y_train)")
    print("  - build_neural_network(input_dim, layers, dropout_rate, learning_rate)")
    print("  - create_early_stopping(patience, monitor, mode)")
    print("  - train_model_with_class_weights(...)")
    print("  - get_model_config_name(layers, dropout, lr)")
