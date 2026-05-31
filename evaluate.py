"""
Evaluate trained weights on a hold-out split of keypoint_data.

Loads isl_model.h5 produced by train.py and reports sklearn accuracy.
For precision/recall/F1, extend with sklearn.classification_report.
"""

import os
from utils import accuracy, get_data
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model("isl_model.h5")
print(model.summary())

X_test, y_test = get_data(train=False)

accuracy_score = accuracy(model, X_test, y_test)
print(f"\nAccuracy : {accuracy_score}")



