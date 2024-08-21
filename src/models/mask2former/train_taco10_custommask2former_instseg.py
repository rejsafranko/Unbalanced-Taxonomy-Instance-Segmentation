from detectron2.config import get_cfg
from detectron2.modeling import META_ARCH_REGISTRY
import CustomMask2FormerHead


def setup_custom_cfg():
    cfg = get_cfg()
    cfg.MODEL.META_ARCHITECTURE = "CustomMask2Former"
    return cfg


# Register the custom Mask2Former head
META_ARCH_REGISTRY.register(CustomMask2FormerHead)
