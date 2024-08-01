import argparse
import json
import torch
import PIL.Image
import open_clip
import sklearn.metrics
from tqdm import tqdm
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to test annotations."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        default="ViT-B-32",
        help="OPEN CLIP pretrained model.",
    )
    return parser.parse_args()


def prepare_text_prompts(path: str) -> Tuple[List[str], dict]:
    with open(path) as f:
        data = json.load(path)

    categories = {category["id"]: category["name"] for category in data["categories"]}
    return list(categories.values()), data


def configure_model(
    model_id: str,
) -> Tuple[
    open_clip.CLIP, open_clip.transform.Compose, open_clip.tokenizer.SimpleTokenizer
]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id, pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_id)
    return model, preprocess, tokenizer


def inference(
    data: dict,
    model: open_clip.CLIP,
    preprocess: open_clip.tokenizer.SimpleTokenizer,
    text: torch.LongTensor,
):
    ground_truths = []
    predictions = []

    for annotation in tqdm(data["annotations"]):
        image_id = annotation["image_id"]
        image_path = f"path/to/images/{image_id:012d}.jpg"
        image_label = annotation["category_id"]

        image = preprocess(PIL.Image.open(image_path).convert("RGB")).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predicted_label = text_probs.argmax(dim=-1).item()

        ground_truths.append(image_label)
        predictions.append(predicted_label)

    return ground_truths, predictions


def calculate_metrics(true_labels: List[int], predicted_labels: List[int]):
    accuracy = sklearn.metrics.accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main(args) -> None:
    text_prompts, taco_data = prepare_text_prompts(args.test_path)
    model, preprocess, tokenizer = configure_model(args.model_id)
    text = tokenizer(text_prompts)
    ground_truths, predictions = inference(taco_data, model, preprocess, text)
    calculate_metrics(true_labels=ground_truths, predicted_labels=predictions)


if __name__ == "__main__":
    args = parse_args()
    main(args)
