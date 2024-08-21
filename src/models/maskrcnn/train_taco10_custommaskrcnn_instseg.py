import models.maskrcnn.CustomROIHeads as CustomROIHeads
from detectron2.config import get_cfg
from detectron2.modeling import ROI_HEADS_REGISTRY

def setup_custom_cfg():
    cfg = get_cfg()
    # Add or modify the relevant configurations
    cfg.MODEL.ROI_HEADS.NAME = "CustomROIHeads"
    return cfg

# Register the custom ROIHeads
ROI_HEADS_REGISTRY.register(CustomROIHeads)