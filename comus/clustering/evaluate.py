import logging
import os

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from comus.clustering.load import get_raw_dataset, get_sal_masks
from comus.clustering.utils import (
    compute_matching,
    get_pixel_accuracy,
    get_prediction,
    one_label_mask,
)
from comus.datasets.coco import COCO_CLASS_NAMES
from comus.datasets.pascal_voc import VOC_CLASSES
from thirdparty.utils import get_iou

N_JOBS = 30

log = logging.getLogger(__name__)


def evaluate_clustering(dataset_results_dir, labels, indexes, cfg):
    """Evaluate provided filtered lables + saliency masks as semantic segmentation method.

    Args:
        dataset_results_dir : dir path to store the results
        pseudo_labels : filtered labels
        pseudo_indexes : filtered indexes
        compute_labels : function to compute pixel_lables
        cfg : hydra config

    Returns:
        pd.DataFrame : DataFrame with mIoU results of the evaluation
    """

    predictions, img_ids = compute_predictions_from_saliency(labels, indexes, cfg.dataset)
    dataset_raw = get_raw_dataset(cfg.dataset)
    predictions, gt_labels = combine_predictions(predictions, img_ids, dataset_raw)
    #  +1 for one additional background class
    match = compute_matching(
        predictions,
        gt_labels,
        cfg.evaluation.matching,
        cfg.clustering.params.n_clusters,
        cfg.dataset.n_classes + 1,
    )

    eval_result = evaluate_predictions(
        predictions,
        gt_labels,
        cfg.dataset.n_classes + 1,
        match,
        dataset_results_dir,
        os.path.join(dataset_results_dir, "results_detailed.csv"),
        cfg.evaluation.matching,
    )

    results = [
        {
            "dataset_name": cfg.dataset.name,
            "n_classes": cfg.dataset.n_classes,
            "model_name": cfg.features.model,
            "clustering_name": cfg.clustering.name,
            "masks_type": cfg.dataset.saliency.type,
            "n_examples": indexes.shape[0],
            "matching": cfg.evaluation.matching,
            "mIoU": eval_result["mIoU"],
        }
    ]
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(dataset_results_dir, "results.csv"))
    return results


def compute_predictions_from_saliency(labels, indexes, dataset_config):
    sal_masks, indexes, img_ids = get_sal_masks(dataset_config, indexes=indexes)
    predictions = [one_label_mask(sal_mask, label) for (sal_mask, label) in zip(sal_masks, labels)]
    return predictions, img_ids


def combine_predictions(predictions, img_ids, dataset_raw, max_size=700):
    # Load all pixel embeddings
    all_pixel_predictions = np.zeros((len(dataset_raw) * max_size * max_size), dtype=np.float32)
    all_gt = np.zeros((len(dataset_raw) * max_size * max_size), dtype=np.float32)
    offset_ = 0
    for idx in range(len(dataset_raw)):
        gt_mask = dataset_raw.get_mask(idx)
        prediction = get_prediction(predictions, idx, dataset_raw, img_ids, gt_mask.shape)

        valid = gt_mask != 255
        n_valid = np.sum(valid)
        all_gt[offset_ : offset_ + n_valid] = gt_mask[valid]

        # Possibly reshape embedding to match gt.
        # assert prediction.shape == gt_mask.shape
        if prediction.shape != gt_mask.shape:
            prediction = cv2.resize(prediction, gt_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
        all_pixel_predictions[offset_ : offset_ + n_valid] = prediction[valid]
        all_gt[offset_ : offset_ + n_valid] = gt_mask[valid]

        # Update offset_
        offset_ += n_valid
    # All pixels, all ground-truth
    all_pixel_predictions = all_pixel_predictions[:offset_]
    all_gt = all_gt[:offset_]
    log.info(f"Found {all_gt.shape[0]} valid pixels.")
    return all_pixel_predictions, all_gt


def evaluate_predictions(
    pixel_predictions,
    pixel_gt_labels,
    n_classes,
    match,
    dataset_results_dir,
    results_file_name,
    matching_name,
    verbose=True,
):
    """Evaluation of flatten predictions.

    Given two vectors of flatten dense GT lables
    and corresponding predictions from unsupervised semantic segmentation with clusters ID
    computes mIoU using provided matching.

    Args:
        pixel_predictions (np.array, Px1): all the pixel predictions (with clustering ID as lables)
        pixel_gt_labels (np.array, Px1): all the pixel lables from original images
        n_classes (int): Number of GT classes
        match (tuple): (CluserID, GT_class) matching
        dataset_results_dir (str): dir path to store the results
        results_file_name (str): file name to store the results
        matching_name (str): type of matching used (hungarian or majority)

    Returns:
        eval_result (dict): dict with mIoU and PA  as well as per category IoU
    """

    log.info("Evaluation of semantic segmentation")

    predictions = np.zeros(pixel_gt_labels.shape[0], dtype=pixel_predictions.dtype)
    for pred_i, target_i in match:
        predictions[pixel_predictions == int(pred_i)] = int(target_i)

    iou = Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
        delayed(get_iou)(predictions, pixel_gt_labels, i_part, i_part) for i_part in range(n_classes)
    )
    iou = np.array(iou)
    pixel_accuracy = get_pixel_accuracy(predictions, pixel_gt_labels)

    eval_result = {
        "per_class": iou,
        "mIoU": np.mean(iou),
        "PA": pixel_accuracy,
    }

    data_path = os.path.join(dataset_results_dir, f"results_{matching_name}.npz")
    np.savez(data_path, **eval_result)

    log.info("Pixel Accuracy is %.2f" % (100 * eval_result["PA"]))
    log.info("Mean IoU is %.2f" % (100 * eval_result["mIoU"]))

    if n_classes == 21:
        class_names = VOC_CLASSES
    elif n_classes == 81:
        class_names = ["background"] + COCO_CLASS_NAMES
    else:
        raise ValueError(f"Not valid number of classes: {n_classes}.")
    data_f = {
        "Classes": ["mIoU", "PA"] + class_names,
        "IoU": [eval_result["mIoU"], eval_result["PA"]] + eval_result["per_class"].tolist(),
    }
    data_path = os.path.join(dataset_results_dir, f"{results_file_name}")
    log.info(dataset_results_dir)
    log.info(data_path)
    pd.DataFrame(data_f).to_csv(data_path)
    if verbose:
        for i_part in range(n_classes):
            log.info(f"Class {class_names[i_part]} has IoU {100 * iou[i_part]:.2f}")
    return eval_result


def save_pseudolabels_saliency(labels, indexes, pseudolabels_dir, cfg):
    """
    Dump fileted lables and indexes to the disk.

    Indexes are used later to restore original images.
    Args:
        pseudo_labels : filtered labels
        pseudo_indexes : filtered indexes
        size : persentage of the core samples, from 0 to 100.
        cfg : hydra config
    """

    log.info(f"Dataset size: {labels.shape[0]}")
    log.info(f"Saving {cfg.clustering.core_size}% of points closest to the cluster centers.")
    pseudolabels_path = os.path.join(pseudolabels_dir, "pseudolabels.npz")

    log.info(f"Saving clustering pseudolabels to {pseudolabels_path}")
    np.savez(pseudolabels_path, pseudolabels=labels, indexes=indexes)
    sal_masks, indexes, img_ids = get_sal_masks(cfg.dataset, indexes=indexes)

    log.info(f"Saving clustering pseudomasks to {pseudolabels_dir}")
    np.save(os.path.join(pseudolabels_dir, "ids.npy"), img_ids)
    for (sal_mask, label, img_id) in zip(sal_masks, labels, img_ids):
        prediction = one_label_mask(sal_mask, label)
        file_name = img_id.split("/")[-1] if isinstance(img_id, str) else f"{img_id:07}"
        mask_file = os.path.join(pseudolabels_dir, f"{file_name}.npy")
        np.save(mask_file, prediction)
