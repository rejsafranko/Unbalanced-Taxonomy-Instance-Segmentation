from argparse import ArgumentParser

from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from mask2former import (
    MaskFormerInstanceDatasetMapper,
    InstanceSegEvaluator,
    add_maskformer2_config,
)

from Trainers import Mask2FormerTrainer

def parse_args():
    pass


def configure_model(config_file:str):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def main(args):
    cfg = configure_model(args.config_file)
    trainer = Mask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__=="__main__":
    args = parse_args()
    main(args)