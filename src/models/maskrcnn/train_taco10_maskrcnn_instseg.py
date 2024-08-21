import argparse
import detectron2  # type: ignore
from Trainers import MaskRCNNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()


def register_taco_dataset(data_dir_path: str):
    for split in ["train", "val", "test"]:
        detectron2.data.datasets.register_coco_instances(
            f"taco10_{split}",
            {},
            data_dir_path + f"annotations_0_{split}.json",
            data_dir_path + "images/",
        )


def configure_model():
    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(
        detectron2.model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = (f"taco10_train",)
    cfg.DATASETS.TEST = (f"taco10_val",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(
        "models/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    return cfg


def main(args: argparse.Namespace):
    register_taco_dataset(data_dir_path=args.data_dir_path, mapping_id=args.mapping)
    cfg = configure_model(num_classes=args.num_classes)
    trainer = MaskRCNNTrainer(cfg)
    trainer.train()
    checkpointer = detectron2.checkpoint.Checkpointer(trainer.model, save_dir="models/")
    checkpointer.save(f"taco10_maskrcnn")


if __name__ == "__main__":
    args = parse_args()
    main(args)
