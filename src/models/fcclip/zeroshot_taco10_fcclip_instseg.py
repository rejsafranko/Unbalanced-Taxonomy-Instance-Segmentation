import argparse
import pickle

import os
import torch
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.evaluation import COCOEvaluator
from fcclip import (
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    add_maskformer2_config,
    add_fcclip_config,
)

PROMPTS = [
    "A discarded {} found on the ground.",
    "A {} lying among scattered trash.",
    "A {} left as litter in the street.",
    "A photo of a {} thrown away in a public space.",
    "A crumpled {} lying near other garbage.",
    "A {} mixed with other debris on the ground.",
    "A broken {} discarded in the environment.",
    "This is a {} among other littered objects.",
    "A small {} found discarded in a cluttered area.",
    "A {} partially covered by other trash in the scene.",
    "A large {} lying near a pile of garbage.",
    "A weathered {} discarded on the side of the road.",
    "A crushed {} among other waste on the street.",
    "A {} that has been thrown away and left as litter.",
    "A {} carelessly discarded in a public park.",
    "A {} found among a variety of litter in this scene.",
    "A {} thrown into the corner of a littered area.",
    "This is a {} abandoned as trash in the environment.",
    "A {} lying in a heap of litter, partially hidden.",
    "A photo of a {} found on the sidewalk among other debris."
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to test annotations."
    )
    return parser.parse_args()


def register_coco_dataset(name, json_file, image_root) -> None:
    detectron2.data.DatasetCatalog.register(
        name,
        lambda: detectron2.data.datasets.load_coco_json(json_file, image_root, name),
    )
    detectron2.data.MetadataCatalog.get(name).set(
        thing_classes=[
            "Can",
            "Other",
            "Bottle",
            "Bottle cap",
            "Cup",
            "Lid",
            "Plastic bag + wrapper",
            "Pop tab",
            "Straw",
            "Cigarette",
        ]
    )


def configure_model():
    cfg = detectron2.config.get_cfg()
    detectron2.projects.deeplab.add_deeplab_config(cfg)
    fcclip.add_maskformer2_config(cfg)
    fcclip.add_fcclip_config(cfg)
    cfg.merge_from_file(
        "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_coco.yaml"
    )
    cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/instseg/fcclip_cocopan_r50.pth"
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 10
    cfg.DATASETS.TEST = ("taco10_test",)
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg


def main(args):
    register_coco_dataset(
        name="taco10_test",
        json_file=f"{args.data_path}mapped_annotations_0_test.json",
        image_root=f"{args.data_path}images/",
    )

    cfg = configure_model()
    predictor = detectron2.engine.DefaultPredictor(cfg)

    detectron2.data.MetadataCatalog.get("taco10_test").set(
        json_file=f"{args.data_path}mapped_annotations_0_test.json"
    )
    
    evaluator = InstanceSegEvaluator("taco10_test",output_dir="./output")

    test_loader = build_detection_test_loader(cfg, dataset_name="taco10_test", mapper=MaskFormerInstanceDatasetMapper(cfg, is_train=False))

    detectron2.evaluation.inference_on_dataset(predictor.model, test_loader, evaluator)

    with open("logs/inference_results.pkl", "rb") as f:
        results = pickle.load(f)

    print("Evaluation Results:", results)
