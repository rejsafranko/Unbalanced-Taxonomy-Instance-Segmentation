import torch
import DynamicWeightedCELoss
from detectron2.modeling.meta_arch.mask2former.mask2former import Mask2FormerHead


class CustomMask2FormerHead(Mask2FormerHead):
    def __init__(self, *args, **kwargs):
        super(CustomMask2FormerHead, self).__init__(*args, **kwargs)
        self.dynamic_loss_fn = DynamicWeightedCELoss(
            num_classes=self.num_classes, device=self.device
        )

    def losses(self, outputs, targets):
        """
        Override the loss computation in Mask2FormerHead to use dynamic weighted loss.
        """
        # Assuming the outputs contain classification logits
        class_logits = outputs["pred_logits"]
        gt_classes = torch.cat([tgt["labels"] for tgt in targets])

        # Compute dynamic weighted cross-entropy loss
        loss_cls = self.dynamic_loss_fn(class_logits, gt_classes)

        # Keep other losses the same
        loss_mask = self.mask_loss(outputs, targets)
        loss_dice = self.dice_loss(outputs, targets)

        return {"loss_cls": loss_cls, "loss_mask": loss_mask, "loss_dice": loss_dice}
