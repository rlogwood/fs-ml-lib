"""
Notebook Setup Module

This module provides standardized setup for Jupyter notebooks including:
- All standard library imports (numpy, pandas, etc.)
- Deep learning imports (TensorFlow, Keras, etc.)
- Custom lib module imports
- Random seed initialization for reproducibility
- Environment information display

IMPORTANT: This module is only available AFTER the bootstrap process completes.

Usage in notebooks:
    # Step 1: Bootstrap (downloads lib if in Colab)
    BOOTSTRAP_URL = 'https://raw.githubusercontent.com/rlogwood/fs-ml-lib/main/colab_bootstrap.py'
    import urllib.request
    exec(urllib.request.urlopen(BOOTSTRAP_URL).read().decode())
    upload_lib(force_refresh=False)

    # Step 2: Setup imports (now lib is available)
    from lib.notebook_setup import setup_notebook
    setup_notebook()

    # Step 3: Enable PyCharm IntelliSense (optional, but recommended for IDE users)
    need_pycharm_intellisense = False
    if need_pycharm_intellisense:
        from lib.notebook_stubs import *

Custom Import Profiles:
    You can create a .notebook_imports.json file to customize imports per notebook:
    {
        "skip_tensorflow": false,
        "skip_sklearn": false,
        "additional_imports": [
            "import torch",
            "from transformers import pipeline"
        ]
    }
"""

import os
import sys
import json
from types import ModuleType


def get_imported_modules(tu: ModuleType):
    """
    Return a human-readable summary of all imported modules and their versions.

    Returns:
    --------
    str : Formatted string listing all imported modules with versions where available
    """
    import sys

    modules_info = []
    modules_info.append(tu.bold_text("IMPORTED MODULES SUMMARY"))

    # Standard imports
    modules_info.append("Standard Libraries:")
    for name in ['numpy', 'pandas', 'os', 'sys', 'zipfile', 'warnings', 'json']:
        if name in sys.modules:
            version = getattr(sys.modules[name], '__version__', 'built-in')
            modules_info.append(f"  ✓ {name:20} {version}")

    # Deep Learning
    modules_info.append("\nDeep Learning:")
    for name in ['tensorflow', 'keras']:
        if name in sys.modules:
            version = getattr(sys.modules[name], '__version__', 'N/A')
            modules_info.append(f"  ✓ {name:20} {version}")

    # Sklearn
    modules_info.append("\nMachine Learning:")
    if 'sklearn' in sys.modules:
        import sklearn
        modules_info.append(f"  ✓ sklearn              {sklearn.__version__}")

    # Visualization
    modules_info.append("\nVisualization:")
    for name in ['matplotlib', 'seaborn', 'PIL']:
        if name in sys.modules:
            version = getattr(sys.modules[name], '__version__', 'N/A')
            modules_info.append(f"  ✓ {name:20} {version}")

    # Custom lib modules
    modules_info.append("\nCustom lib modules:")
    lib_modules = sorted([name for name in sys.modules if name.startswith('lib.')])
    for name in lib_modules:
        modules_info.append(f"  ✓ {name}")

    #modules_info.append("=" * 70)

    return "\n".join(modules_info)


def load_import_config(config_file='.notebook_imports.json'):
    """
    Load custom import configuration from a JSON file.

    Parameters:
    -----------
    config_file : str, optional
        Path to the import configuration file (default: .notebook_imports.json)

    Returns:
    --------
    dict : Configuration dictionary with import settings
    """
    default_config = {
        'skip_tensorflow': False,
        'skip_sklearn': False,
        'skip_visualization': False,
        'additional_imports': []
    }

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                print(f"✓ Loaded custom import config from {config_file}")
        except Exception as e:
            print(f"⚠ Error loading {config_file}: {e}")

    return default_config


def setup_notebook(show_versions=True, show_imports=False, config_file=None):
    """
    Complete notebook setup including all imports and configuration.

    This function handles:
    - All standard and deep learning imports
    - Custom lib module imports and reloading
    - Random seed initialization for reproducibility
    - Optional version information display
    - Custom import profiles via config file

    Parameters:
    -----------
    show_versions : bool, optional (default=True)
        If True, prints TensorFlow, Keras, and GPU availability information
    show_imports : bool, optional (default=False)
        If True, prints detailed list of all imported modules after setup
    config_file : str, optional (default=None)
        Path to custom import configuration JSON file

    Example:
    --------
    >>> setup_notebook()
    >>> # Now use imports directly
    >>> np.random.rand(5)
    >>> tu.print_heading("My Heading")
    >>> plt.plot([1, 2, 3])

    >>> # Show all imports for documentation
    >>> setup_notebook(show_imports=True)

    >>> # Use custom import profile
    >>> setup_notebook(config_file='.notebook_imports.json')
    """
    # Load custom configuration if specified
    config = load_import_config(config_file) if config_file else {
        'skip_tensorflow': False,
        'skip_sklearn': False,
        'skip_visualization': False,
        'additional_imports': []
    }
    # Standard imports
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import zipfile

    # Collect all modules for injection
    modules_to_inject = {
        'np': np,
        'pd': pd,
        'os': os,
        'zipfile': zipfile,
        'warnings': warnings
    }

    # Deep Learning imports (conditional)
    if not config.get('skip_tensorflow', False):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                             GlobalAveragePooling2D, Flatten, Input)
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50

        modules_to_inject.update({
            'tf': tf,
            'keras': keras,
            'Sequential': Sequential,
            'Model': Model,
            'Dense': Dense,
            'Dropout': Dropout,
            'BatchNormalization': BatchNormalization,
            'GlobalAveragePooling2D': GlobalAveragePooling2D,
            'Flatten': Flatten,
            'Input': Input,
            'Adam': Adam,
            'EarlyStopping': EarlyStopping,
            'ReduceLROnPlateau': ReduceLROnPlateau,
            'ModelCheckpoint': ModelCheckpoint,
            'ImageDataGenerator': ImageDataGenerator,
            'VGG16': VGG16,
            'MobileNetV2': MobileNetV2,
            'ResNet50': ResNet50
        })

    # Sklearn imports (conditional)
    if not config.get('skip_sklearn', False):
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        modules_to_inject.update({
            'confusion_matrix': confusion_matrix,
            'classification_report': classification_report,
            'accuracy_score': accuracy_score
        })

    # Visualization (conditional)
    if not config.get('skip_visualization', False):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from PIL import Image
        from IPython.display import display, HTML

        modules_to_inject.update({
            'plt': plt,
            'sns': sns,
            'Image': Image,
            'display': display,
            'HTML': HTML
        })

    # Public git repo fs-ml-lib imports
    # For local development clone the repo from:
    # https://github.com/rlogwood/fs-ml-lib.git
    import lib.text_util as tu
    import lib.wrangler as wr
    import lib.data_cleaner as dc
    import lib.analyzer as da
    import lib.corr_analysis as ca
    import lib.utility as utl
    import lib.data_downloader as ddl

    modules_to_inject.update({
        'tu': tu,
        'wr': wr,
        'dc': dc,
        'da': da,
        'ca': ca,
        'utl': utl,
        'ddl': ddl
    })

    # Reload modules after code changes
    from lib.colab_bootstrap import reload_lib_modules
    reload_lib_modules()

    # Set random seeds for reproducibility
    np.random.seed(42)
    if 'tf' in modules_to_inject:
        modules_to_inject['tf'].random.set_seed(42)

    # Process additional imports from config
    if config.get('additional_imports'):
        print(f"\n✓ Processing {len(config['additional_imports'])} additional imports...")
        for import_statement in config['additional_imports']:
            try:
                exec(import_statement, globals(), modules_to_inject)
                print(f"  ✓ {import_statement}")
            except Exception as e:
                print(f"  ✗ Failed: {import_statement} - {e}")

    # Display version information if requested
    if show_versions:
        if 'tf' in modules_to_inject:
            print(f"TensorFlow version: {modules_to_inject['tf'].__version__}")
            print(f"Keras version: {modules_to_inject['keras'].__version__}")
            print(f"GPU Available: {modules_to_inject['tf'].config.list_physical_devices('GPU')}")
        else:
            print("TensorFlow: Skipped")

    # Make imports available in the calling scope
    # This allows direct usage like: np.array(), tu.print_heading(), etc.
    import inspect
    frame = inspect.currentframe().f_back
    frame.f_globals.update(modules_to_inject)

    # Show detailed import list if requested
    if show_imports:
        print("\n" + get_imported_modules(tu=tu))

