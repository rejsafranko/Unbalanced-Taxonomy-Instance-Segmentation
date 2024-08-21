import torch
import torch.nn as nn

class DynamicWeightedCELoss(nn.Module):
    def __init__(self, num_classes, device):
        super(DynamicWeightedCELoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.register_buffer('class_counts', torch.zeros(num_classes, device=device))
        self.register_buffer('class_recalls', torch.zeros(num_classes, device=device))
        
    def update_class_counts(self, gt_classes):
        for cls in range(self.num_classes):
            self.class_counts[cls] += (gt_classes == cls).sum().item()
    
    def update_class_recalls(self, pred_classes, gt_classes):
        for cls in range(self.num_classes):
            tp = ((pred_classes == cls) & (gt_classes == cls)).sum().item()
            fn = ((pred_classes != cls) & (gt_classes == cls)).sum().item()
            recall = tp / (tp + fn + 1e-10)  # To avoid division by zero
            self.class_recalls[cls] = recall
    
    def forward(self, logits, gt_classes):
        self.update_class_counts(gt_classes)
        
        pred_classes = torch.argmax(logits, dim=1)
        self.update_class_recalls(pred_classes, gt_classes)
        
        weights = 1.0 / (self.class_counts + 1e-10)  # Inverse class frequency
        dynamic_weights = weights * (1 - self.class_recalls)
        dynamic_weights = dynamic_weights / dynamic_weights.sum()  # Normalize weights
        
        loss = nn.CrossEntropyLoss(weight=dynamic_weights)(logits, gt_classes)
        return loss