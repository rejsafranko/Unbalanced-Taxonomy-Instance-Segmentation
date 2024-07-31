import argparse
import detectron2 # type: ignore
import fcclip # type: ignore
from Trainers import FCCLIPTrainer


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
    detectron2.projects.deeplab.add_deeplab_config(cfg)
    fcclip.add_maskformer2_config(cfg)
    fcclip.add_fcclip_config(cfg)
    cfg.merge_from_file("models/configs/fcclip_convnext_large_eval_coco.yaml")
    cfg.DATASETS.TRAIN = ("taco10_train",)
    cfg.DATASETS.TEST = ("taco10_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 10
    cfg.MODEL_WEIGHTS = (
        "https://drive.google.com/file/d/1tcB-8FNON-LwckXQbUyKcBA2G7TU65Zh/view"
    )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = 100
    cfg.freeze()
    return cfg


def main(args):
    register_taco_dataset(args.data_path)
    cfg = configure_model()
    trainer = FCCLIPTrainer(cfg)
    trainer.train()
    checkpointer = detectron2.checkpoint.Checkpointer(trainer.model, save_dir="models/")
    checkpointer.save(f"taco10_fcclip")


if __name__ == "__main__":
    main()
