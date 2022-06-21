import collections.abc
import csv
import logging
import os

import albumentations as A
import numpy as np
import torch
import torch.distributed as dist
from albumentations.pytorch import ToTensorV2
from torch import nn
from tqdm import tqdm

from comus.datasets.coco import COCOSegmentationDatasetPseudolabels
from comus.datasets.pascal_voc import PascalVOCSegmentationPseudolabels
from comus.utils import collate_fn_masks
from thirdparty import utils
from thirdparty.segmentation import segmentation_model

log = logging.getLogger(__name__)


def self_training(
    gpu,
    output_dir,
    pseudolabels_dir,
    cfg,
    checkpoint_path=None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=8, rank=gpu)

    if not cfg.self_training.pretrained_backbone:
        log.info("Ablation. Using random initial weights")
    model = segmentation_model(
        "deeplabv3",
        "resnet50",
        num_classes=cfg.evaluation.n_clusters + 1,
        pretrained_backbone=cfg.self_training.pretrained_backbone,
    ).cuda()
    if checkpoint_path is not None:
        to_restore = {"epoch": 0}
        utils.restart_from_checkpoint(
            checkpoint_path, remove_module=False, run_variables=to_restore, model=model
        )
    if cfg.dataset.name == "pascal":
        dataset_train = PascalVOCSegmentationPseudolabels(
            root=cfg.dataset.root,
            n_clusters=cfg.evaluation.n_clusters,
            image_set=cfg.self_training.train_split,
            pseudolables_data_dir=pseudolabels_dir,
            transform=_get_augmentations(cfg.self_training),
        )
    elif cfg.dataset.name in ["coco", "coco_pascal"]:
        dataset_train = COCOSegmentationDatasetPseudolabels(
            root=cfg.dataset.root,
            idx_dir=cfg.dataset.idx_dir,
            split=cfg.self_training.train_split,
            cat_list=cfg.dataset.cat_list,
            pseudolables_data_dir=pseudolabels_dir,
            transform=_get_augmentations(cfg.self_training),
        )
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} is not supported.")
    log.info(f"Dataset train size: {len(dataset_train)}")
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True, rank=gpu)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.self_training.batch_size_per_gpu,
        shuffle=False,
        num_workers=cfg.self_training.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler_train,
        collate_fn=collate_fn_masks,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = _get_optimizer(cfg.self_training, model.parameters())
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = train_model(model, criterion, loader_train, optimizer, output_dir, cfg)


def _get_augmentations(cfg):
    transforms_list = [
        A.RandomResizedCrop(
            width=cfg.input_size,
            height=cfg.input_size,
            scale=cfg.crops_scale,
            interpolation=1,
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(transforms_list, additional_targets={"pseudo_mask": "mask"})


def _get_optimizer(cfg, parameters):
    optmizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
    assert cfg.optimizer.name in optmizers, f"Invalid optimizer {cfg.optimizer}"
    return optmizers[cfg.optimizer.name](parameters, **cfg.optimizer.params)


def train_model(model, criterion, dataloader, optimizer, output_dir, cfg):

    # Optionally resume training
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(output_dir, "checkpoint.pth"),
        remove_module=False,
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ["epoch", "Train_loss", "IoU"]
    with open(os.path.join(output_dir, "log.csv"), "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if start_epoch == 0 and utils.is_main_process():
            writer.writeheader()
    for epoch in range(start_epoch, cfg.self_training.n_epochs + 1):
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}
        model.train()  # Set model to training mode
        for (_, imgs, pseudo_masks) in tqdm(dataloader):
            inputs = imgs.cuda(non_blocking=True)
            # zero the parameter gradients
            optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(
                    outputs["out"],
                    pseudo_masks.cuda(non_blocking=True).long().argmax(dim=1),
                )
                loss.backward()
                optimizer.step()
            batchsummary["epoch"] = epoch
            epoch_loss = loss
            batchsummary["Train_loss"] = epoch_loss.item()
        for field in fieldnames:
            if isinstance(field, collections.abc.Sequence):
                batchsummary[field] = np.mean(batchsummary[field])
        if utils.is_main_process():
            with open(os.path.join(output_dir, "log.csv"), "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(batchsummary)
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": cfg.self_training,
        }
        utils.save_on_master(save_dict, os.path.join(output_dir, "checkpoint.pth"))
        if cfg.self_training.saveckp_freq and epoch % cfg.self_training.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(output_dir, f"checkpoint{epoch:04}.pth"))
    return model
