import argparse
import json
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    return parser.parse_args()


def save_mapped_annotation(data, path: str) -> None:
    save_path = path.split("/")
    save_path = f"{save_path[0]}/{save_path[1]}/mapped_{save_path[2]}"
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


def encode_target_classes(classes):
    return {category: idx + 1  for idx, category in enumerate(classes)}


def encode_original_classes(data):
    return {category["name"]: int(category["id"]) for category in data["categories"]}


def create_target_categories(encoding):
    categories = []
    for category in encoding:
        categories.append(
            {
                "id": encoding[category],
                "name": category,
                "supercategory": category,
            }
        )
    return categories


def encode_class_mappings(mapping, encodings):
    mapping["original_class"] = mapping["original_class"].map(encodings["original"])
    mapping["new_class"] = mapping["new_class"].map(encodings["target"])
    mapping = dict(zip(mapping["original_class"], mapping["new_class"]))
    return mapping


def map_taco_classes(taco_path: str, mapping_path: str):
    class_mapping = pd.read_csv(mapping_path)

    target_encoding = encode_target_classes(class_mapping["new_class"].unique())
    target_categories = create_target_categories(target_encoding)

    with open(taco_path, "r") as f:
        taco_data = json.load(f)

    original_encoding = encode_original_classes(taco_data)

    taco_data["categories"] = target_categories

    class_mapping = encode_class_mappings(
        class_mapping, {"original": original_encoding, "target": target_encoding}
    )

    for annotation in taco_data["annotations"]:
        annotation["category_id"] = class_mapping[annotation["category_id"]]

    save_mapped_annotation(taco_data, taco_path)


def main(args):
    map_taco_classes(args.annotation_path, args.mapping_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
