import logging
import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from skimage.measure import regionprops
from tqdm import tqdm

from comus.saliency import load_sal_model
from comus.utils import collate_fn_masks
from thirdparty.mscoco import COCOSegmentation
from thirdparty.utils import mkdir_if_missing

log = logging.getLogger(__name__)

COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class COCOSegmentationWrapper(COCOSegmentation):
    def __getitem__(self, index):
        img_id = self.ids[index]
        image = super().get_image(index)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return img_id, image, torch.zeros_like(image)


class COCOSegmentationwithMaskWrapper(COCOSegmentation):
    def __getitem__(self, index):
        img_id = self.ids[index]
        image = super().get_image(index)
        mask = super().get_mask(index)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            mask, image = transformed["mask"], transformed["image"]
        return img_id, image, mask


class COCOSegmentationDatasetSaliency(COCOSegmentation):
    def __init__(self, root, idx_dir, cat_list, split, saliency):
        super().__init__(root=root, idx_dir=idx_dir, cat_list=cat_list, split=split, transform=None)
        assert saliency.type in ["supervised", "unsupervised"]
        assert saliency.size in [512, 256, 128, 64], "Only power of 2 size is supported"
        self.original_root = root
        self.sal = saliency
        self.dino_inputs_size = saliency.dino_inputs_size
        name = f"{self.cat_list_name}_{self.split}_sal_{self.sal.type}_{self.sal.size}_{self.sal.th}"
        index_file = os.path.join(self.idx_dir, f"{name}_ids.npy")
        self.sal_dir = os.path.join(self.idx_dir, name)
        mkdir_if_missing(self.sal_dir)
        if os.path.exists(index_file) and saliency.use_saved:
            self.ids = np.load(index_file)
        else:
            log.info(f"Creating new filtered index in {index_file}")
            self.ids = self._create_object_index(index_file)

    def _create_object_index(self, index_file):
        log.info(f"Compute {self.sal.type} from scratch")
        transforms_list = [
            A.Resize(self.sal.size, self.sal.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        transform = A.Compose(transforms_list)
        dataset = COCOSegmentationWrapper(
            root=self.original_root,
            idx_dir=self.idx_dir,
            split=self.split,
            cat_list=self.cat_list_name,
            transform=transform,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.sal.batch_size,
            shuffle=False,
            collate_fn=collate_fn_masks,
        )
        basnet = load_sal_model(self.sal.path, self.sal.type, self.sal.th)
        basnet = torch.nn.DataParallel(basnet)
        all_img_ids = []
        for img_ids, imgs, _ in tqdm(loader):
            sal_masks = basnet(imgs.cuda()).cpu()
            new_ids = self._save_saliency_masks(sal_masks, img_ids)
            all_img_ids.append(new_ids)
        object_index = np.concatenate(all_img_ids)
        np.save(index_file, object_index)
        return object_index

    def _save_saliency_masks(self, sal_masks, img_ids):
        new_ids = []
        for sal_mask, img_id in zip(sal_masks, img_ids):
            coco = self.coco
            img_metadata = coco.loadImgs(img_id)[0]
            sal = cv2.resize(
                sal_mask.numpy().astype(np.int64),
                (img_metadata["width"], img_metadata["height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            regions = regionprops(sal)
            assert len(regions) <= 1
            # filter small salinency masks
            if len(regions) == 1 and regions[0].area > self.sal.min_size:
                region = regions[0]
                bbox = (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2])
                np.savez(os.path.join(self.sal_dir, f"{img_id:07}.npz"), sal=sal, bbox=bbox)
                new_ids.append(img_id)
        new_ids = np.array(new_ids)
        return new_ids

    def get_sal_mask(self, index):
        img_id = self.ids[index]
        mask_path = os.path.join(self.sal_dir, f"{img_id:07}.npz")
        with np.load(mask_path) as mask:
            sal = mask["sal"]
            bbox = mask["bbox"]
        return sal, bbox

    def __getitem__(self, index):
        image = self.get_image(index)
        sal, bbox = self.get_sal_mask(index)
        assert sal.shape == image.shape[:2]
        transforms_list = [
            A.Crop(*bbox),
            A.Resize(self.dino_inputs_size, self.dino_inputs_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        transform = A.Compose(transforms_list)
        transformed = transform(image=image, mask=sal)
        sal, image = transformed["mask"], transformed["image"]
        sal_masks_binary = torch.cat([(sal == False).float()[None], (sal == True).float()[None]])
        return index, image, sal_masks_binary


class COCOSegmentationDatasetPseudolabels(COCOSegmentation):
    def __init__(self, root, idx_dir, cat_list, split, pseudolables_data_dir, transform):
        super().__init__(root, idx_dir, cat_list, split, transform)
        self.pseudo_masks_dir = pseudolables_data_dir
        self.ids = np.load(os.path.join(self.pseudo_masks_dir, "ids.npy"))
        self.n_clusters = 81 if cat_list == "coco_full" else 21

    def get_pseudo_mask(self, index):
        img_id = self.ids[index]
        mask_path = os.path.join(self.pseudo_masks_dir, f"{img_id:07}.npy")
        pseudo_mask = np.load(mask_path)
        pseudo_mask = self._convert_to_segmentation_mask_cluster_lables(pseudo_mask)
        return pseudo_mask

    def _convert_to_segmentation_mask_cluster_lables(self, mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, self.n_clusters), dtype=np.float32)
        for label_index in range(self.n_clusters):
            bin_mask = (mask == label_index).astype(float)
            segmentation_mask[:, :, label_index] = bin_mask
        return segmentation_mask

    def __getitem__(self, index):
        image = self.get_image(index)
        pseudo_mask = self.get_pseudo_mask(index)
        assert pseudo_mask.shape[:2] == image.shape[:2]
        if self.transform is not None:
            transformed = self.transform(image=image, pseudo_mask=pseudo_mask)
            image = transformed["image"]
            pseudo_mask = transformed["pseudo_mask"].permute(2, 0, 1)
        return index, image, pseudo_mask
