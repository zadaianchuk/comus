import os
import shutil

import torch

from thirdparty.utils import mkdir_if_missing


def update_dirs(dataset_results_dir, i):
    output_dir = os.path.join(dataset_results_dir, f"results_{i}")
    pseudolabels_dir = os.path.join(dataset_results_dir, f"pseudolables_{i}")
    prev_output_dir = os.path.join(dataset_results_dir, f"results_{i-1}")
    prev_pseudolabels_dir = os.path.join(dataset_results_dir, f"pseudolables_{i-1}")
    if os.path.exists(prev_pseudolabels_dir):
        shutil.rmtree(prev_pseudolabels_dir)
    mkdir_if_missing(output_dir)
    mkdir_if_missing(pseudolabels_dir)
    return output_dir, pseudolabels_dir, prev_output_dir


def collate_fn_masks(batch):

    index = []
    images = []
    masks = []

    for sample in batch:
        index.append(sample[0])
        images.append(sample[1])
        masks.append(sample[2])

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    return index, images, masks
