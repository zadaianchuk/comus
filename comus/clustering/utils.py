import logging

import numpy as np

from thirdparty.utils import _hungarian_match, _majority_vote

log = logging.getLogger(__name__)

N_JOBS = 30


def compute_matching(pixel_predictions, pixel_gt_labels, matching, number_clusters, n_classes):
    if matching == "majority":
        log.info("Using majority voting for matching with IoU as score.")
        match = _majority_vote(
            pixel_predictions,
            pixel_gt_labels,
            preds_k=number_clusters + 1,
            targets_k=n_classes,
        )
    elif matching == "hungarian":
        log.info("Using hungarian matching with IoU as score.")
        match = _hungarian_match(
            pixel_predictions,
            pixel_gt_labels,
            preds_k=number_clusters + 1,
            targets_k=n_classes,
        )
    else:
        raise NotImplementedError
    return match


def one_label_mask(sal_mask, label):
    mask = np.zeros_like(sal_mask).astype(np.int32)
    mask[sal_mask.astype(np.bool)] = label + 1
    return mask


def get_prediction(predictions, idx, dataset_raw, img_ids, shape):
    img_ids = list(img_ids)
    if dataset_raw.ids[idx] in img_ids:
        return predictions[img_ids.index(dataset_raw.ids[idx])]
    else:
        return np.zeros(shape)


def get_pixel_accuracy(flat_preds, flat_targets):
    tp = (flat_preds == flat_targets).sum()
    return float(tp) / flat_preds.shape[0]
