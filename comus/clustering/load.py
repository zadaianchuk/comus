import logging

import numpy as np
import torch
import torch.optim
import torch.utils.data
from torch import nn

from comus.datasets.coco import COCOSegmentation, COCOSegmentationDatasetSaliency
from comus.datasets.pascal_voc import PascalVOCSegmentation, PascalVOCSegmentationSaliency
from comus.utils import collate_fn_masks

log = logging.getLogger(__name__)


def load_model(model_name, data_parallel=False):
    """Load pretrained self-supervised DINO weights to models."""
    assert model_name in [
        "dino_vits8",
        "dino_vitb8",
        "dino_vitb16",
        "dino_vits16",
    ], "{model_name} representations are not supported"

    model = torch.hub.load("facebookresearch/dino:main", model_name)
    model.fc = nn.Identity()
    model = model.cuda()
    if data_parallel:
        model = torch.nn.DataParallel(model)
    return model


def get_dataset_loader(cfg):
    if cfg.dataset.name == "pascal":
        dataset = PascalVOCSegmentationSaliency(
            root=cfg.dataset.root,
            split=cfg.dataset.split,
            saliency=cfg.dataset.saliency,
        )
    elif cfg.dataset.name in ["coco", "coco_pascal"]:
        dataset = COCOSegmentationDatasetSaliency(
            root=cfg.dataset.root,
            idx_dir=cfg.dataset.idx_dir,
            split=cfg.dataset.split,
            cat_list=cfg.dataset.cat_list,
            saliency=cfg.dataset.saliency,
        )
    else:
        raise NotImplementedError
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        collate_fn=collate_fn_masks,
        num_workers=40,
    )
    return loader, dataset, dataset.classes


def get_raw_dataset(dataset_config):
    if dataset_config.name == "pascal":
        dataset_raw = PascalVOCSegmentation(
            root=dataset_config.root, split=dataset_config.split, transform=None
        )
    elif dataset_config.name in ["coco", "coco_pascal"]:
        dataset_raw = COCOSegmentation(
            split=dataset_config.split,
            root=dataset_config.root,
            idx_dir=dataset_config.idx_dir,
            cat_list=dataset_config.cat_list,
            transform=None,
        )
    else:
        raise NotImplementedError
    return dataset_raw


def get_sal_masks(dataset_config, indexes):
    # to load already saved saliency
    dataset_config.saliency.use_saved = True
    if dataset_config.name == "pascal":
        dataset_sal = PascalVOCSegmentationSaliency(
            root=dataset_config.root,
            split=dataset_config.split,
            saliency=dataset_config.saliency,
        )

    elif dataset_config.name in ["coco", "coco_pascal"]:
        dataset_sal = COCOSegmentationDatasetSaliency(
            root=dataset_config.root,
            idx_dir=dataset_config.idx_dir,
            split=dataset_config.split,
            cat_list=dataset_config.cat_list,
            saliency=dataset_config.saliency,
        )
    else:
        raise NotImplementedError
    img_ids = np.array([dataset_sal.ids[ind] for ind in indexes])
    sal_masks = (dataset_sal.get_sal_mask(ind)[0] for ind in indexes)
    return sal_masks, indexes, img_ids
