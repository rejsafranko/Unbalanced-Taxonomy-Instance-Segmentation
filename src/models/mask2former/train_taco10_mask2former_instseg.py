import argparse
import detectron2  # type: ignore
import mask2former  # type: ignore

from Trainers import Mask2FormerTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()


def register_taco_dataset(data_path: str) -> None:
    for split in ["train", "val", "test"]:
        detectron2.data.datasets.register_coco_instances(
            f"taco10_{split}",
            {},
            data_path + f"mapped_annotations_0_{split}.json",
            data_path + "images/",
        )


def configure_model(config_file: str):
    """
    Create configs and perform basic setups.
    """
    cfg = detectron2.config.get_cfg()
    detectron2.projects.deeplab.add_deeplab_config(cfg)
    mask2former.add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TRAIN = ("taco10_train",)
    cfg.DATASETS.TEST = ("taco10_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 10
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.STEPS = []
    return cfg


def main(args: argparse.Namespace) -> None:
    register_taco_dataset(args.data_path)
    cfg = configure_model(args.config_file)
    trainer = Mask2FormerTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
