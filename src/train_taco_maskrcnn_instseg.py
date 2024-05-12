from argparse import ArgumentParser
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultEngine
from detectron2.utils.logger import setup_logger
import json
import numpy as np
import os
import random

# Custom imports.
from Trainer import Trainer

setup_logger()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=False, default=60)
    return parser.parse_args()


def register_taco_dataset(data_dir_path: str, num_classes: int):
    for split in ["train", "val", "test"]:
        register_coco_instances(
            f"taco{str(num_classes)}_{split}",
            {},
            data_dir_path + f"annotations_0_{split}.json",
            data_dir_path + "images/",
        )


def configure_model(num_classes: int):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = (f"taco{str(num_classes)}_train",)
    cfg.DATASETS.TEST = (f"taco{str(num_classes)}_val",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    return cfg


def main(args):
    register_taco_dataset(data_dir_path=args.data_dir_path, mapping_id=args.mapping)
    cfg = configure_model(num_classes=args.num_classes)
    trainer = Trainer(cfg)
    trainer.train()
    checkpointer = DetectionCheckpointer(trainer.model, save_dir="models/")
    checkpointer.save(f"taco{args.num_classes}_maskrcnn")

if __name__ == "__main__":
    args = parse_args()
    main(args)
