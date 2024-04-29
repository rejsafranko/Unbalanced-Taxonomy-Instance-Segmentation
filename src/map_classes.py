"""

Author: Pedro F. Proenza

"""

import argparse
import copy
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--class_map_csv", type=str, required=True)
    return parser.parse_args()


class Taco:
    def load_taco(self, dataset_dir, subset, class_map_csv, round=0):
        class_map = {}
        # map_to_one_class = {}
        with open(class_map_csv) as csvfile:
            reader = csv.reader(csvfile)
            class_map = {row[0]: row[1] for row in reader}
            # map_to_one_class = {c: 'Litter' for c in class_map}

        ann_filepath = os.path.join(dataset_dir, "annotations")
        ann_filepath += "_" + str(round) + "_" + subset + ".json"
        assert os.path.isfile(ann_filepath)

        # Load dataset.
        dataset = json.load(open(ann_filepath, "r"))

        # Replace dataset original classes before calling the coco Constructor.
        # Some classes may be assigned background to remove them from the dataset.
        self.replace_dataset_classes(dataset, class_map)

    def replace_dataset_classes(self, dataset, class_map):
        """Replaces classes of dataset based on a dictionary"""
        class_new_names = list(set(class_map.values()))
        class_new_names.sort()
        class_originals = copy.deepcopy(dataset["categories"])
        dataset["categories"] = []
        class_ids_map = {}  # map from old id to new id

        # Assign background id 0.
        has_background = False
        if "Background" in class_new_names:
            if class_new_names.index("Background") != 0:
                class_new_names.remove("Background")
                class_new_names.insert(0, "Background")
            has_background = True

        # Replace categories.
        for id_new, class_new_name in enumerate(class_new_names):

            # Make sure id:0 is reserved for background.
            id_rectified = id_new
            if not has_background:
                id_rectified += 1

            category = {
                "supercategory": "",
                "id": id_rectified,  # Background has id=0
                "name": class_new_name,
            }
            dataset["categories"].append(category)
            # Map class names
            for class_original in class_originals:
                if class_map[class_original["name"]] == class_new_name:
                    class_ids_map[class_original["id"]] = id_rectified

        # Update annotations category id tag
        for ann in dataset["annotations"]:
            ann["category_id"] = class_ids_map[ann["category_id"]]


def main(args):
    taco_client = Taco()
    taco_client.load_taco(
        dataset_dir=args.dataset_dir, subset=args.subset, class_map_csv=args.class_map_csv
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
