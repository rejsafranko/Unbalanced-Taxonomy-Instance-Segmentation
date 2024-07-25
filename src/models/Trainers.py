import os
import detectron2 # type: ignore
import mask2former # type: ignore


class MaskRCNNTrainer(detectron2.engine.DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskRCNN.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return detectron2.evaluator.COCOEvaluator(
            dataset_name, cfg, True, output_folder
        )


class Mask2FormerTrainer(detectron2.engine.DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return detectron2.evaluator.COCOEvaluator(
            dataset_name, cfg, True, output_folder
        )

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = mask2former.MaskFormerInstanceDatasetMapper(cfg, True)
            return detectron2.data.build_detection_train_loader(cfg, mapper=mapper)
