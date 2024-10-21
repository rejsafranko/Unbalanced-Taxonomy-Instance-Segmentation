import torch
import torch.nn.functional as F


class Mask2FormerLossBalancer:
    def __init__(self, class_frequencies, total_samples, eos_file=0.1):
        self.class_frequencies = class_frequencies
        self.total_samples = total_samples
        class_frequencies_tensor = torch.tensor(
            list(self.class_frequencies.values()), dtype=torch.float32
        )
        self.class_weights = self.total_samples / class_frequencies_tensor
        self.eos_file = eos_file
        self.class_weights = torch.cat(
            [self.class_weights, torch.tensor([self.eos_file])]
        )

    def update_weights(self, recalls):
        class_frequencies_tensor = torch.tensor(
            list(self.class_frequencies.values()), dtype=torch.float32
        )
        base_weights = self.total_samples / class_frequencies_tensor
        recall_factors = 1.0 - recalls
        class_weights = (
            self.class_weights.clone()
        )  # Clone to avoid modifying the original tensor.
        class_weights[
            :-1
        ] *= recall_factors  # Apply the recall factor only to object classes.
        self.class_weights = class_weights


    def loss_function(self, predictions, targets):
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            raise ValueError("Predictions contain NaN or Inf values.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise ValueError("Targets contain NaN or Inf values.")

        dynamic_weights = self.class_weights
        dynamic_weights = dynamic_weights.to(targets.device)
        class_weights = dynamic_weights[targets.long()]

        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, weight=class_weights, reduction="none"
        )
        return loss
