import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    # When imported as part of a package
    from . import text_util as tu
except ImportError:
    # When run as a standalone script
    import text_util as tu


def inspect_variable(var, var_name="variable"):
    print(f"=== {var_name} ===")
    print(f"Class: {var.__class__}")
    print(f"Class Name: {var.__class__.__name__}")
    print(f"Type: {type(var)}")
    print(f"Type name: {type(var).__name__}")
    print(f"String representation: {str(var)[:100]}...")
    if hasattr(var, 'shape'):
        print(f"Shape: {var.shape}")
    print()


def show_breakdown(df, col1, col2, from_val: float = None, to_val: float = None):
    def print_summary(title):
        tu.print_sub_heading(title)
        # summary = df.groupby(col1)[col2].value_counts().unstack(fill_value=0)

        if from_val is not None:
            if to_val is not None:
                # print(summary.loc[from_val:to_val,:])
                print(tu.bold_text(f'Breakdown between {from_val} and {to_val}'))
                print(summary.loc[(summary.index > from_val) & (summary.index <= to_val)])
                print(tu.italic_text(tu.bold_text('Total')))
                print(summary.loc[(summary.index > from_val) & (summary.index <= to_val)].sum())
            else:
                print(tu.bold_text(f'Breakdown from {from_val}'))
                print(summary.loc[(summary.index > from_val)])
                print(tu.italic_text(tu.bold_text('Total')))
                print(summary.loc[(summary.index > from_val)].sum())
        else:
            print(summary)

    tu.print_heading(f'summary of {col1} vs {col2}')
    summary = pd.crosstab(df[col1], df[col2])
    print_summary('crosstab')

    # summary = df.groupby(col1)[col2].value_counts().unstack(fill_value=0)
    # print_summary('group_by')
    #
    # summary = df.pivot_table(index=col1, columns=col2, aggfunc='size', fill_value=0)
    # print_summary('pivot_table')


def show_env():
    import os
    from textwrap import wrap

    env_vars = dict(os.environ)

    print("\n" + "=" * 100)
    print(f"{'ENVIRONMENT VARIABLES':^100}")
    print("=" * 100)
    print(f"Total: {len(env_vars)} variables\n")

    for key in sorted(env_vars.keys()):
        value = env_vars[key]

        # Wrap long values across multiple lines with indentation
        if len(value) > 80:
            wrapped = wrap(value, width=80)
            print(f"\033[1m{key}\033[0m:")
            for line in wrapped:
                print(f"  {line}")
            print()
        else:
            print(f"\033[1m{key:<35}\033[0m = {value}")

    print("=" * 100 + "\n")


def get_predictions(model, X, threshold=0.5, verbose=0):
    """
    Get class predictions and probabilities from any model type.

    This function works with both sklearn-style classifiers (with predict_proba)
    and Keras/TensorFlow models (where predict returns probabilities).

    Parameters:
    -----------
    model : object
        Trained classifier model (sklearn or Keras)
    X : array-like
        Input features for prediction
    threshold : float, default=0.5
        Probability threshold for binary classification
    verbose : int, default=0
        Verbosity level for Keras models (0=silent, 1=progress bar)

    Returns:
    --------
    y_pred : ndarray
        Binary class predictions (0 or 1)
    y_pred_proba : ndarray
        Probability of positive class (1D array)

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> y_pred, y_proba = get_predictions(model, X_test)

    >>> from tensorflow import keras
    >>> model = keras.Sequential([...])
    >>> model.fit(X_train, y_train)
    >>> y_pred, y_proba = get_predictions(model, X_test, verbose=0)
    """
    if hasattr(model, 'predict_proba'):
        # sklearn-style model with probability support
        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
        except (TypeError, IndexError):
            # Mock object or non-standard predict_proba - fall back to predict
            try:
                y_pred_proba = model.predict(X, verbose=verbose).flatten()
            except TypeError:
                y_pred_proba = model.predict(X).flatten()
    else:
        # Keras/TensorFlow model - predict returns probabilities
        try:
            # Try with verbose parameter (Keras models)
            y_pred_proba = model.predict(X, verbose=verbose).flatten()
        except TypeError:
            # Fall back without verbose (sklearn or other models)
            y_pred_proba = model.predict(X).flatten()

    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    return y_pred, y_pred_proba


@dataclass
class ModelArchitectureInfo:
    """
    Generic model architecture and training information.

    This dataclass provides a structured way to describe any model type
    (Keras, sklearn, XGBoost, etc.) with consistent attributes, including
    both architecture details and optional training configuration.

    Attributes:
    -----------
    model_type : str
        Class name of the model (e.g., "Sequential", "RandomForestClassifier")
    model_family : str
        Broad category of model: "neural_network", "tree_ensemble",
        "linear", "gradient_boosting", or "unknown"
    n_parameters : int, optional
        Total number of trainable parameters (neural networks)
    n_layers : int, optional
        Number of layers (neural networks)
    input_shape : tuple, optional
        Input shape for neural networks
    config : dict, optional
        Key hyperparameters and configuration settings
    layers : list, optional
        List of layer descriptions for neural networks
    batch_size : int, optional
        Training batch size (if available)
    epochs : int, optional
        Number of training epochs (if available)
    learning_rate : float, optional
        Learning rate used during training (if available)

    Examples:
    ---------
    >>> info = ModelArchitectureInfo(
    ...     model_type="Sequential",
    ...     model_family="neural_network",
    ...     n_parameters=1234,
    ...     n_layers=5,
    ...     batch_size=32,
    ...     epochs=100
    ... )
    >>> print(info)
    Sequential (neural_network), 5 layers, 1,234 parameters, trained: 100 epochs, batch_size=32
    """
    model_type: str
    model_family: str
    n_parameters: Optional[int] = None
    n_layers: Optional[int] = None
    input_shape: Optional[tuple] = None
    config: Optional[Dict[str, Any]] = None
    layers: Optional[list] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None

    def layer_summary(self):
        summary = ""
        for layer in self.layers:
            name = layer.get('name','')
            type = layer.get('type','')
            units = layer.get('units','')
            output_shape = layer.get('output_shape','')
            activation = layer.get('activation','')
            rate = layer.get('rate','')

            summary += f"  - {name} ({type}:{units}:{output_shape}) {activation} {rate}\n"
        return summary


    def __str__(self) -> str:
        """
        Human-readable description of the model architecture and training config.

        Returns:
        --------
        str
            Formatted string describing the model
        """
        parts = [f"{self.model_type} ({self.model_family})"]
        if self.n_parameters:
            parts.append(f"{self.n_parameters:,} parameters")

        if self.n_layers:
            parts.append(f"{self.n_layers} layers")
            parts.append(f"\n{self.layer_summary()}")

        if self.config:
            # Show up to 3 key config items
            config_items = [(k, v) for k, v in self.config.items() if v is not None][:3]
            if config_items:
                config_str = ', '.join(f"{k}={v}" for k, v in config_items)
                parts.append(config_str)

        # Add training configuration if available
        training_parts = []
        if self.epochs is not None:
            training_parts.append(f"{self.epochs} epochs")
        if self.batch_size is not None:
            training_parts.append(f"batch_size={self.batch_size}")
        if self.learning_rate is not None:
            training_parts.append(f"lr={self.learning_rate}")

        if training_parts:
            parts.append(f"trained: {', '.join(training_parts)}")

        return ', '.join(parts)


def add_batch_size_to_history(history, batch_size):
    """
    Add batch_size to history.params for later extraction.

    Keras doesn't store batch_size in history.params by default, so this
    helper function adds it manually for use with get_model_architecture_info().

    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object returned by model.fit()
    batch_size : int
        The batch size used during training

    Returns:
    --------
    keras.callbacks.History
        The same history object with batch_size added to params

    Usage:
    ------
    >>> history = model.fit(X, y, batch_size=32, epochs=100)
    >>> history = add_batch_size_to_history(history, 32)
    >>> info = get_model_architecture_info(model, history=history)
    """
    if history is not None:
        params_dict = getattr(history, 'params', None)
        if params_dict and isinstance(params_dict, dict):
            params_dict['batch_size'] = batch_size
    return history


def debug_history_params(history):
    """
    Debug helper to see what's in the history object.

    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object

    Usage:
    ------
    >>> history = model.fit(X, y, batch_size=32, epochs=100)
    >>> debug_history_params(history)
    """
    print("\n" + "=" * 70)
    print("DEBUG: History object contents")
    print("=" * 70)

    if history is None:
        print("History is None")
        return

    print(f"\nType: {type(history)}")

    if hasattr(history, 'params'):
        print(f"\nhistory.params keys: {list(history.params.keys())}")
        print("\nhistory.params contents:")
        for key, value in history.params.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo 'params' attribute found")

    if hasattr(history, 'epoch'):
        print(f"\nhistory.epoch: {history.epoch}")
        print(f"Number of epochs trained: {len(history.epoch)}")
    else:
        print("\nNo 'epoch' attribute found")

    if hasattr(history, 'history'):
        print(f"\nhistory.history keys: {list(history.history.keys())}")

    print("=" * 70)


def get_model_architecture_info(model, history=None, batch_size=None, epochs=None, learning_rate=None) -> ModelArchitectureInfo:
    """
    Extract architecture and training information from any model type.

    This function works with Keras/TensorFlow models, sklearn models,
    XGBoost models, and provides generic fallback for other types.

    Parameters:
    -----------
    model : object
        Trained model (Keras, sklearn, XGBoost, etc.)
    history : keras.callbacks.History, optional
        Training history object (Keras models). If provided, extracts batch_size and epochs automatically.
    batch_size : int, optional
        Training batch size (overrides history if provided)
    epochs : int, optional
        Number of training epochs (overrides history if provided)
    learning_rate : float, optional
        Learning rate (if not provided, auto-extracted from Keras optimizer)

    Returns:
    --------
    ModelArchitectureInfo
        Structured model information with type, family, configuration, and training params

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(n_estimators=100, max_depth=10)
    >>> info = get_model_architecture_info(model)
    >>> print(info)
    RandomForestClassifier (tree_ensemble), n_estimators=100, max_depth=10

    >>> from tensorflow import keras
    >>> model = keras.Sequential([...])
    >>> history = model.fit(X, y, batch_size=32, epochs=100)
    >>> info = get_model_architecture_info(model, history=history)
    >>> print(info)
    Sequential (neural_network), 5 layers, 1,234 parameters, trained: 100 epochs, batch_size=32, lr=0.001
    """
    # Extract training params from history if available and not explicitly provided
    if history is not None:
        # Check if history has params attribute (it should be a dict)
        params_dict = getattr(history, 'params', None)

        if params_dict and isinstance(params_dict, dict):
            # Try to extract batch_size from history.params
            if batch_size is None:
                # First try direct batch_size key (added by train_model_with_class_weights)
                if 'batch_size' in params_dict:
                    batch_size = params_dict['batch_size']
                # Otherwise calculate from 'steps' and 'samples' if available
                elif 'samples' in params_dict and 'steps' in params_dict:
                    samples = params_dict['samples']
                    steps = params_dict['steps']
                    if steps > 0:
                        batch_size = int(samples / steps)

        # Try to extract epochs from history.epoch (actual epochs trained - respects early stopping)
        if epochs is None:
            if hasattr(history, 'epoch') and history.epoch:
                epochs = len(history.epoch)
            # Fallback to params if epoch list not available
            elif params_dict and 'epochs' in params_dict:
                epochs = params_dict['epochs']

    # Auto-extract learning rate from Keras optimizer if not provided
    if learning_rate is None and hasattr(model, 'optimizer'):
        try:
            if hasattr(model.optimizer, 'learning_rate'):
                lr_value = model.optimizer.learning_rate
                # Handle TensorFlow Variable
                if hasattr(lr_value, 'numpy'):
                    learning_rate = float(lr_value.numpy())
                else:
                    learning_rate = float(lr_value)
        except (AttributeError, TypeError):
            pass  # Couldn't extract learning rate, leave as None

    model_type = type(model).__name__

    # Keras/TensorFlow models
    if hasattr(model, 'layers'):
        config = {}
        if hasattr(model, 'optimizer'):
            config['optimizer'] = model.optimizer.__class__.__name__
        if hasattr(model, 'loss'):
            loss_name = model.loss if isinstance(model.loss, str) else model.loss.__class__.__name__
            config['loss'] = loss_name

        # Extract layer information
        layer_descriptions = []
        for layer in model.layers:
            # Extract meaningful output dimension (last element of shape tuple, excluding batch dimension)
            output_dim = None
            if hasattr(layer, 'output_shape') and layer.output_shape:
                # output_shape is typically (None, dim) or (None, dim1, dim2, ...)
                # We want the last non-None dimension
                shape_tuple = layer.output_shape
                if isinstance(shape_tuple, tuple) and len(shape_tuple) > 1:
                    output_dim = shape_tuple[-1]  # Last dimension (e.g., 32 from (None, 32))

            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': output_dim,
            }

            # Add units for Dense layers
            if hasattr(layer, 'units'):
                layer_info['units'] = layer.units

            # Add activation for layers that have it
            if hasattr(layer, 'activation') and layer.activation is not None:
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                layer_info['activation'] = activation_name

            # Add dropout rate
            if hasattr(layer, 'rate'):
                layer_info['rate'] = layer.rate

            layer_descriptions.append(layer_info)

        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='neural_network',
            n_parameters=model.count_params() if hasattr(model, 'count_params') else None,
            n_layers=len(model.layers),
            input_shape=model.input_shape if hasattr(model, 'input_shape') else None,
            config=config if config else None,
            layers=layer_descriptions,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )

    # sklearn tree-based models
    elif 'Forest' in model_type or 'Tree' in model_type:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        config = {
            k: v for k, v in {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'min_samples_split': params.get('min_samples_split')
            }.items() if v is not None
        }
        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='tree_ensemble',
            config=config if config else None,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )

    # sklearn linear models
    elif 'Logistic' in model_type or 'Linear' in model_type or 'Ridge' in model_type:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        config = {
            k: v for k, v in {
                'C': params.get('C'),
                'penalty': params.get('penalty'),
                'solver': params.get('solver')
            }.items() if v is not None
        }
        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='linear',
            config=config if config else None,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )

    # XGBoost models
    elif 'XGB' in model_type:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        config = {
            k: v for k, v in {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('learning_rate')
            }.items() if v is not None
        }
        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='gradient_boosting',
            config=config if config else None,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )

    # Generic fallback for unknown model types
    else:
        config = None
        if hasattr(model, 'get_params'):
            all_params = model.get_params()
            # Try to extract a few interesting parameters
            interesting_keys = ['n_estimators', 'max_depth', 'C', 'alpha', 'learning_rate']
            config = {k: v for k, v in all_params.items() if k in interesting_keys and v is not None}

        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='unknown',
            config=config if config else None,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
