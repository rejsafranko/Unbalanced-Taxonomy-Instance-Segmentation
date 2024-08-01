import argparse
import pickle
import detectron2  # type: ignore
from modules.fcclip.fcclip import FCCLIP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to test annotations."
    )
    return parser.parse_args()


def register_coco_dataset(name, json_file, image_root):
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
    cfg.merge_from_file("models/configs/fcclip_convnext_large_eval_coco.yaml")
    cfg.MODEL.WEIGHTS = (
        "https://drive.google.com/file/d/1tcB-8FNON-LwckXQbUyKcBA2G7TU65Zh/view"
    )
    cfg.MODEL.FC_CLIP.TEST.INSTANCE_ON = True
    cfg.DATASETS.TEST = ("taco10_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = "cuda"
    return cfg


def main(args):
    register_coco_dataset(
        name="taco10_test",
        json_file=f"{args.data_path}mapped_annotations_0_test.json",
        image_root=f"{args.data_path}images/",
    )

    cfg = configure_model()
    model = FCCLIP(cfg)
    model.eval()

    predictor = detectron2.engine.DefaultPredictor(cfg)
    test_loader = detectron2.data.build_detectron_test_loader(cfg, "taco10_test")
    evaluator = detectron2.evaluation.COCOEvaluator(
        "taco10_test", cfg, False, output_dir="logs/"
    )
    detectron2.evaluation.inference_on_dataset(predictor.model, test_loader, evaluator)

    with open("logs/inference_results.pkl", "rb") as f:
        results = pickle.load(f)

    print("Evaluation Results:", results)
