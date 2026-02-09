"""
Type stubs for notebook imports - Use this for PyCharm IntelliSense

Add this cell to your notebook AFTER setup_notebook() to get IDE support:

    # Type hints for PyCharm (never executes, only for IDE)
    need_pycharm_intellisense = False
    if need_pycharm_intellisense:
        from lib.notebook_stubs import *

Why use a variable instead of `if False:`?
- PyCharm's static analyzer treats literal `if False:` as dead code and may skip it
- Using a variable forces the analyzer to evaluate the branch and load type information
- This enables full IntelliSense for os, zipfile, np, pd, tf, tu, etc.
- At runtime, the condition is False, so the import never executes
- Actual imports are already injected by setup_notebook() via inspect.currentframe()

This pattern gives you the best of both worlds:
✓ Centralized import management via setup_notebook()
✓ Full PyCharm IntelliSense and code completion
✓ Zero runtime overhead (condition is False, imports already cached)
"""

# Standard library
import os
import sys
import zipfile
import warnings
import json

# Data science
import numpy as np
import pandas as pd

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50

# Machine Learning
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import display, HTML

# Custom lib modules
import lib.text_util as tu
import lib.wrangler as wr
import lib.data_cleaner as dc
import lib.analyzer as da
import lib.corr_analysis as ca
import lib.utility as utl
import lib.data_downloader as ddl

# Export all for 'from notebook_stubs import *'
__all__ = [
    'os', 'sys', 'zipfile', 'warnings', 'json',
    'np', 'pd',
    'tf', 'keras', 'Sequential', 'Model',
    'Dense', 'Dropout', 'BatchNormalization',
    'GlobalAveragePooling2D', 'Flatten', 'Input',
    'Adam', 'EarlyStopping', 'ReduceLROnPlateau', 'ModelCheckpoint',
    'ImageDataGenerator', 'VGG16', 'MobileNetV2', 'ResNet50',
    'confusion_matrix', 'classification_report', 'accuracy_score',
    'plt', 'sns', 'Image', 'display', 'HTML',
    'tu', 'wr', 'dc', 'da', 'ca', 'utl', 'ddl'
]
