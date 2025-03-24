import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from IAWF1_metric import IAWF1

confusion_matrix_example = np.array([[99999, 1],   
                                     [9999999, 1]]) 

# Experiment 1: Using manual weights
manual_weights = [0.7, 0.3]  # Custom weights for the classes
metric_manual = IAWF1(custom_weights=manual_weights)
scores_manual, weights_manual = metric_manual.score(confusion_matrix_example)
print(f"IAWF1 Score (Manual Weights): {scores_manual}, Weights Used: {weights_manual}")

# Experiment 2: Using automatic weights
metric_auto = IAWF1()  # No custom weights provided
scores_auto, weights_auto = metric_auto.score(confusion_matrix_example)
print(f"IAWF1 Score (Automatic Weights): {scores_auto}, Weights Used: {weights_auto}")


