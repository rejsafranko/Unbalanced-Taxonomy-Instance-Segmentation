import os

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluator import COCOEvaluator
from mask2former import MaskFormerInstanceDatasetMapper


class MaskRCNNTrainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskRCNN.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

class Mask2FormerTrainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)