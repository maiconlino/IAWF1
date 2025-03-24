import numpy as np

class IAWF1:
    """
    Implementation of the IAWF1 metric for evaluating models on imbalanced test datasets.

    Attributes:
    - custom_weights: Optional manual weights for each class. If None, automatic weights are used.
    """
    def __init__(self, custom_weights=None):
        self.custom_weights = np.array(custom_weights, dtype=float) if custom_weights else None

    def score(self, confusion_matrix):
        """
        Compute the IAWF1 metric given a confusion matrix and user-defined or automatic weights.

        Parameters:
        - confusion_matrix: Numpy array of shape (n_classes, n_classes), representing the confusion matrix.

        Returns:
        - IAWF1_score: Computed IAWF1 metric score.
        - final_weights: The weights applied to each class (automatic or user-defined).
        """
        # Extracting values from the confusion matrix
        TN = confusion_matrix[0, 0]  # True Negatives
        FP = confusion_matrix[0, 1]  # False Positives
        FN = confusion_matrix[1, 0]  # False Negatives
        TP = confusion_matrix[1, 1]  # True Positives

        # Step 1: Initialize variables
        f1_scores = np.zeros(2)  # For two classes (survivors and deaths)
        class_sizes = np.zeros(2)  # To store class sizes

        # Step 2: Calculate F1-scores
        precision_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        recall_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        f1_scores[0] = (2 * precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
        class_sizes[0] = TN + FN

        precision_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
        recall_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_scores[1] = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
        class_sizes[1] = TP + FP

        # Step 3: Define weights
        if self.custom_weights is not None:
            final_weights = self.custom_weights
        else:
            final_weights = class_sizes / np.sum(class_sizes) if np.sum(class_sizes) > 0 else np.array([0.5, 0.5])

        # Step 4: Compute IAWF1 score
        IAWF1_score = np.sum(final_weights * f1_scores)

        return IAWF1_score, final_weights

