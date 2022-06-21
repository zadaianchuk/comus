import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from comus.clustering.evaluate import combine_predictions, compute_matching, evaluate_predictions
from comus.datasets.coco import COCOSegmentation, COCOSegmentationWrapper
from comus.datasets.pascal_voc import PascalVOCSegmentation, PascalVOCSegmentationWrapper
from comus.utils import collate_fn_masks
from thirdparty.segmentation import segmentation_model
from thirdparty.utils import restart_from_checkpoint


def load_data(cfg, split):
    transforms_list = [
        A.Resize(cfg.self_training.input_size, cfg.self_training.input_size, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    transform = A.Compose(transforms_list, additional_targets={"pseudo_mask": "mask"})
    if cfg.dataset.name in ["pascal"]:
        dataset_raw = PascalVOCSegmentation(split=split, root=cfg.dataset.root, transform=None)
        dataset = PascalVOCSegmentationWrapper(
            split=split, root=cfg.dataset.root, transform=transform
        )
        n_classes = 21
    elif cfg.dataset.name in ["coco", "coco_pascal"]:
        dataset_raw = COCOSegmentation(
            root=cfg.dataset.root,
            idx_dir=cfg.dataset.idx_dir,
            split=split,
            cat_list=cfg.dataset.cat_list,
            transform=None,
        )
        dataset = COCOSegmentationWrapper(
            root=cfg.dataset.root,
            idx_dir=cfg.dataset.idx_dir,
            split=split,
            cat_list=cfg.dataset.cat_list,
            transform=transform,
        )
        n_classes = 81 if cfg.dataset.name == "coco" else 21
    else:
        raise ValueError(f"{cfg.dataset.name} is not valid. Pick from 'coco' or 'pascal'.")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.eval_model.batch_size * cfg.eval_model.n_gpus,
        shuffle=False,
        num_workers=cfg.eval_model.num_workers,
        collate_fn=collate_fn_masks,
    )
    return loader, dataset_raw, n_classes


def load_model(checkpoint_path, cfg):
    model = segmentation_model(
        "deeplabv3",
        "resnet50",
        num_classes=cfg.evaluation.n_clusters + 1,
        pretrained_backbone=True,
    ).cuda()

    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        checkpoint_path, remove_module=True, run_variables=to_restore, model=model
    )
    if cfg.eval_model.n_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    return model


def eval_model(checkpoint_path, results_dir, cfg):
    loader, dataset_raw, n_classes = load_data(cfg, split=cfg.eval_model.val_split)
    model = load_model(checkpoint_path, cfg)

    predictions, img_ids = compute_predictions(model, loader, cfg)
    predictions, gt_labels = combine_predictions(predictions, img_ids, dataset_raw)
    match = compute_matching(
        pixel_predictions=predictions,
        pixel_gt_labels=gt_labels,
        matching=cfg.evaluation.matching,
        number_clusters=cfg.evaluation.n_clusters,
        n_classes=n_classes,
    )
    np.save(
        os.path.join(results_dir, f"{cfg.evaluation.matching}_match.npy"),
        np.asarray(match),
    )
    evaluate_predictions(
        pixel_predictions=predictions,
        pixel_gt_labels=gt_labels,
        n_classes=n_classes,
        match=match,
        dataset_results_dir=results_dir,
        results_file_name=os.path.join(
            results_dir, f"results_{cfg.evaluation.matching}_detailed.csv"
        ),
        matching_name=cfg.evaluation.matching,
    )


def compute_predictions(model, loader, cfg):
    predictions = []
    img_ids_all = []
    for img_ids, imgs, _ in tqdm(loader, "Masks inference"):
        inputs = imgs.cuda()
        outputs = model(inputs)["out"].argmax(dim=1).cpu().numpy()
        predictions.append(outputs)
        img_ids_all.append(img_ids)
    img_ids = np.concatenate(img_ids_all)
    predictions = np.concatenate(predictions, axis=0)
    return predictions, img_ids


def save_pseudolables_model(pseudolables_dir, checkpoint_path, cfg):
    model = load_model(checkpoint_path, cfg)
    loader, dataset_raw, _ = load_data(cfg, split=cfg.eval_model.pseudolables_split)

    img_ids_all = []

    for img_ids, imgs, _ in tqdm(loader, "Masks saving"):
        inputs = imgs.cuda()
        predictions = model(inputs)["out"].argmax(dim=1).cpu().numpy()
        for (prediction, img_id) in zip(predictions, img_ids):
            file_name = img_id.split("/")[-1] if type(img_id) == str else f"{img_id:07}"
            image_dims = dataset_raw.get_shape(img_id)
            prediction = cv2.resize(prediction, image_dims[::-1], interpolation=cv2.INTER_NEAREST)
            mask_file = os.path.join(pseudolables_dir, f"{file_name}.npy")
            np.save(mask_file, prediction)
        img_ids_all.append(img_ids)
    img_ids = np.concatenate(img_ids_all)
    np.save(os.path.join(pseudolables_dir, "ids.npy"), img_ids)
