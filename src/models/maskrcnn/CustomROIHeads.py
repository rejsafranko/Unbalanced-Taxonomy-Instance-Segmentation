from detectron2.modeling.roi_heads import StandardROIHeads

class CustomROIHeads(StandardROIHeads):
    def _forward_box(self, features, proposals):
        box_features = self.box_pooler([features[f] for f in self.in_features], [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, {}