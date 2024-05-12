import os
from argparse import ArgumentParser
from typing import List

from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from coco_types.dicts import coco_object_detection


def parse_args():
    """
    Parse command line arguments for the script. Arguments include the directory to save masks and the path to the annotation files.

    Returns:
        Namespace: Command line arguments with keys 'mask_dir' for mask directory and 'ann_path' for annotations path.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory where the masks will be saved.",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        required=True,
        help="Base path to annotation files with split names appended before '.json'.",
    )
    return parser.parse_args()


def load_annotations(
    imgId: int, catIds:int, coco: COCO, iscrowd=None
) -> List[coco_object_detection.Annotation]:
    """
    Load annotations for a given image ID from a COCO dataset.

    Args:
        imgIds (int): ID of the image to load annotations for.
        coco (COCO): COCO object for the dataset.
        iscrowd (Optional[bool]): Selector for crowd annotations. Default is None, which includes all annotations.

    Returns:
        List[dict]: List of annotations.
    """
    ann_ids = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=iscrowd)
    anns = coco.loadAnns(ann_ids)
    return anns


def generate_mask(
    annotations: List[coco_object_detection.Annotation],
    img_height: int,
    img_width: int,
    coco: COCO,
) -> np.ndarray:
    """
    Generate a binary mask from COCO annotations for an image.

    Args:
        annotations (List[dict]): List of annotation dictionaries for the image.
        img_height (int): The height of the image.
        img_width (int): The width of the image.
        coco (COCO): COCO dataset object.

    Returns:
        np.ndarray: Binary mask array where pixels of objects are 1 and others are 0.
    """
    mask = np.zeros((img_height, img_width))
    
    for ann in annotations:
        try:
            mask_ = coco.annToMask(ann)
            mask += mask_
        except ValueError as e:
            print(f"Error adding masks: {e}\nNum. of annotations: {len(annotations)}\nMask shape: {mask_.shape}")

    return mask


def save_mask(mask: np.ndarray, file_name: str, mask_dir: str) -> None:
    """
    Save a mask image to disk.

    Args:
        mask (np.ndarray): The mask array to save.
        file_name (str): Original image file name to base the mask's file name on.
        mask_dir (str): Directory where the mask image will be saved.
    """
    mask_img = Image.fromarray((mask).astype(np.uint8))
    mask_img.save(os.path.join(mask_dir, f"{file_name.split('/')[1][:-4]}_mask.png"))


def main(args) -> None:
    """
    Main function to generate masks for images in specified COCO format JSON annotations for train, validation, and test splits.

    Args:
        args (Namespace): Parsed command line arguments containing 'mask_dir' and 'ann_path'.
    """
    for split in ["train", "val", "test"]:
        coco_split = COCO(args.ann_path + split + ".json")
        cat_ids = coco_split.getCatIds()
        images = coco_split.imgs
        for img in images.values():
            anns = load_annotations(imgId=img["id"], catIds=cat_ids, coco=coco_split)
            mask = generate_mask(
                annotations=anns,
                img_height=img["height"],
                img_width=img["width"],
                coco=coco_split,
            )
            save_mask(mask=mask, file_name=img["file_name"], mask_dir=args.mask_dir + split)
        print(f"All {split} masks have been generated and saved.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
