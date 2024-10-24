import numpy as np
import itertools
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import COCOEvaluator
from tabulate import tabulate


class CustomCOCOEvaluator(COCOEvaluator):
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type)
            + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        recalls = coco_eval.eval["recall"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        ar_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            recall = recalls[:, idx, 0, -1]
            precision = precision[precision > -1]
            recall = recall[recall > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            ar = np.mean(recall) if recall.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            ar_per_category.append(("{}".format(name), float(ar * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})

        # tabulate AR
        ar_flatten = list(itertools.chain(*ar_per_category))
        ar_2d = itertools.zip_longest(*[ar_flatten[i::N_COLS] for i in range(N_COLS)])
        ar_table = tabulate(
            ar_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AR: \n".format(iou_type) + ar_table)

        results.update({"AR-" + name: ar for name, ar in ar_per_category})

        return results
