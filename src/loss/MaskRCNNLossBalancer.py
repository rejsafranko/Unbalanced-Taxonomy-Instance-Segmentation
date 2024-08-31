import torch
import torch.nn.functional as F


class MaskRCNNLossBalancer:
    def __init__(self, class_frequencies, total_samples):
        self.class_frequencies = class_frequencies
        self.total_samples = total_samples
        class_frequencies_tensor = torch.tensor(
            list(self.class_frequencies.values()), dtype=torch.float32
        )
        self.class_weights = self.total_samples / class_frequencies_tensor

    def update_weights(self, recalls):
        """
        Ova metoda ažurira težine na temelju trenutnog `recall` za svaku klasu.
        """
        class_frequencies_tensor = torch.tensor(
            list(self.class_frequencies.values()), dtype=torch.float32
        )
        base_weights = self.total_samples / class_frequencies_tensor
        recall_factors = 1.0 - recalls
        self.class_weights = base_weights * recall_factors

    def loss_function(self, predictions, targets):
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            raise ValueError("Predictions contain NaN or Inf values.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise ValueError("Targets contain NaN ili Inf values.")

        if self.class_weights is None:
            raise ValueError(
                "Class weights have not been initialized. Call update_weights first."
            )

        dynamic_weights = self.class_weights.to(targets.device)
        class_weights = dynamic_weights[targets.long()]

        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, weight=class_weights, reduction="mean"
        )
        return loss
