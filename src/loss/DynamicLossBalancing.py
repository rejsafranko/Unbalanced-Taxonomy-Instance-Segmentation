import torch
import torch.nn.functional as F


class DynamicLossBalancing:
    def __init__(self, class_frequencies):
        # Calculate weights based on class frequencies
        class_frequencies_tensor = torch.tensor(
            list(class_frequencies.values()), dtype=torch.float32
        )
        self.class_weights = 1.0 / class_frequencies_tensor
        self.class_weights /= self.class_weights.sum()  # Normalize

    def dynamic_weighting(self, recall, max_recall_factor=10):
        # Adjust weights based on recall
        recall_factor = 1.0 / (recall + 1e-8)
        recall_factor = min(recall_factor, max_recall_factor)
        dynamic_weights = self.class_weights * recall_factor
        return dynamic_weights

    def loss_function(self, predictions, targets, recall):
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            raise ValueError("Predictions contain NaN or Inf values.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise ValueError("Targets contain NaN or Inf values.")
        # Compute dynamic weights
        dynamic_weights = self.dynamic_weighting(recall)
        dynamic_weights = dynamic_weights.to(targets.device)
        # Convert dynamic weights to match target dimensions
        class_weights = dynamic_weights[targets.long()]

        # Apply weights to Binary Cross Entropy Loss with logits
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, weight=class_weights, reduction="mean"
        )
        return loss
