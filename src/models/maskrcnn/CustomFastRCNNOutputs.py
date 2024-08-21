import torch
import models.maskrcnn.DynamicWeightedCELoss as DynamicWeightedCELoss
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class CustomFastRCNNOutputs(FastRCNNOutputLayers):
    def __init__(self, *args, **kwargs):
        super(CustomFastRCNNOutputs, self).__init__(*args, **kwargs)
        self.dynamic_loss_fn = DynamicWeightedCELoss(num_classes=self.num_classes, device=self.cls_score.weight.device)
    
    def losses(self, predictions, proposals):
        scores, bbox_deltas = predictions
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        loss_cls = self.dynamic_loss_fn(scores, gt_classes)
        
        loss_box_reg = torch.nn.functional.smooth_l1_loss(
            bbox_deltas,
            torch.cat([p.gt_boxes.tensor for p in proposals]),
            self.smooth_l1_beta,
            reduction="sum",
        )
        
        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg / gt_classes.numel()}