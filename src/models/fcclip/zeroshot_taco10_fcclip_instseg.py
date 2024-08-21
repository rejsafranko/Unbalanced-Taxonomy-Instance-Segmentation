import argparse
import pickle

import fcclip

import detectron2  # type: ignore
import detectron2.engine  # type: ignore
import detectron2.data # type: ignore
import detectron2.evaluation  # type: ignore
import detectron2.data  # type: ignore
import detectron2.projects.deeplab # type: ignore
from detectron2.evaluation import COCOEvaluator  # type: ignore


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
    test_loader = detectron2.data.build_detection_test_loader(
        cfg,
        dataset_name="taco10_test",
        mapper=fcclip.MaskFormerInstanceDatasetMapper(cfg, is_train=False),
    )
    evaluator = detectron2.evaluation.COCOEvaluator(
        "taco10_test", output_dir="./output"
    )
    detectron2.evaluation.inference_on_dataset(predictor.model, test_loader, evaluator)

    with open("logs/inference_results.pkl", "rb") as f:
        results = pickle.load(f)

    print("Evaluation Results:", results)
