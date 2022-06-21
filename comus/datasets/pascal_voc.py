import logging
import os
from typing import List

import albumentations as A
import cv2
import imageio
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage.measure import regionprops
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from comus.saliency import load_sal_model
from comus.utils import collate_fn_masks

log = logging.getLogger(__name__)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

IM_SIZE = 256
transforms_list = [
    A.Resize(IM_SIZE, IM_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]

TRANSFORM = A.Compose(transforms_list, additional_targets={"pseudo_mask": "mask"})

transforms_list_without_normalization = [
    A.Resize(IM_SIZE, IM_SIZE),
    ToTensorV2(),
]

TRANSFORM_NO_NORM = A.Compose(
    transforms_list_without_normalization, additional_targets={"pseudo_mask": "mask"}
)


class PascalVOCSegmentation(VisionDataset):
    classes = VOC_CLASSES
    _SPLITS_DIR = "Segmentation"
    _TARGET_FILE_EXT = ".png"

    def __init__(self, root, split="val", transform=TRANSFORM):
        if split in ["trainaug_one", "trainaug_many", "val_one", "val_many"]:
            self.filter = split.split("_")[-1]
            split = split.split("_")[0]
        else:
            self.filter = None
        self.transform = transform
        self.root = root
        self.year = 2012
        self._TARGET_DIR = "SegmentationClass"
        if split == "trainaug":
            self._TARGET_DIR = "SegmentationClassAug"
        base_dir = "VOCdevkit/VOC2007" if split == "test" else "VOCdevkit/VOC2012"
        voc_root = os.path.join(self.root, base_dir)

        if not os.path.isdir(voc_root):
            raise RuntimeError(f"{voc_root} Dataset not found or corrupted.")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, split.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, f"{x}.jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        if split == "trainaug":
            self.root = root
            self._convert_to_segmentation_mask = self._convert_to_segmentation_mask_lables
            TARGET_DIR = "SegmentationClassAug"
            voc_root = os.path.join(self.root, "VOCdevkit/VOC2012")
            splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
            split_f = os.path.join(splits_dir, split.rstrip("\n") + ".txt")
            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]

            image_dir = os.path.join(voc_root, "JPEGImages")
            self.images = [os.path.join(image_dir, f"{x}.jpg") for x in file_names]

            target_dir = os.path.join(voc_root, TARGET_DIR)
            self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

            assert len(self.images) == len(self.targets)
        self.file_names = file_names

    @property
    def ids(self):
        return self.images

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        """Converts a mask from the Pascal VOC format to the format required by AutoAlbument."""
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)

        return segmentation_mask

    @staticmethod
    def _convert_to_segmentation_mask_lables(mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, _ in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label_index, axis=-1).astype(float)
        return segmentation_mask

    def __len__(self) -> int:
        return len(self.images)

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)

        return index, image, mask

    def get_shape(self, img_id):
        image = cv2.imread(img_id)
        return image.shape[:2]

    def get_mask(self, index):
        mask = np.array(Image.open(self.masks[index]))
        return mask


class PascalVOCSegmentationWrapper(PascalVOCSegmentation):
    def __getitem__(self, index):
        _, image, mask = super().__getitem__(index)
        return self.images[index], image, mask


class PascalVOCSegmentationSaliency(PascalVOCSegmentation):
    def __init__(self, root, split, saliency):
        super().__init__(root=root, split=split, transform=None)
        self.sal = saliency
        self.dino_inputs_size = saliency.dino_inputs_size
        if saliency.type != "dir":
            self.sal_masks = self.get_saliency_masks(
                dataset_root=root, dataset_split=split, sal=saliency
            )
        self.images = self._create_object_index()

    def _create_object_index(self):
        filtered_sal_masks = []
        filtered_images = []
        filtered_bbox = []
        if self.sal.type != "dir":
            for image_path, mask in zip(self.images, self.sal_masks):
                image = cv2.imread(image_path)
                mask = cv2.resize(
                    mask.astype(np.int32),
                    image.shape[:2][::-1],
                    interpolation=cv2.INTER_NEAREST,
                )
                if mask.sum() != 0:
                    regions = regionprops(mask)
                    assert len(regions) == 1
                    region = regions[0]
                    # we filter saliency masks smaller than 20x20 pixels
                    # as their crop resize representation is meaningless
                    if region.area > self.sal.min_size:
                        bbox = (
                            region.bbox[1],
                            region.bbox[0],
                            region.bbox[3],
                            region.bbox[2],
                        )
                        filtered_sal_masks.append(mask)
                        filtered_images.append(image_path)
                        filtered_bbox.append(bbox)
            log.info(f"Found {len(filtered_images)} images with valid saliency masks.")
            self.sal_masks = filtered_sal_masks
            self.bbox = filtered_bbox
        else:
            filtered_images = self.images

        return filtered_images

    @staticmethod
    def get_saliency_masks(dataset_root, dataset_split, sal):
        name = f"{sal.type}_sal_masks_bin_th_{sal.th}_{dataset_split}"
        data_path = os.path.join(dataset_root, f"{name}.npz")
        if not sal.use_saved or not os.path.isfile(data_path):
            log.info(f"Compute {sal.type} from scratch")
            transforms_list = [
                A.Resize(sal.size, sal.size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
            transform = A.Compose(transforms_list)
            dataset = PascalVOCSegmentation(
                root=dataset_root, split=dataset_split, transform=transform
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=sal.batch_size,
                shuffle=False,
                collate_fn=collate_fn_masks,
            )
            basnet = load_sal_model(sal.path, sal.type, sal.th)
            basnet = torch.nn.DataParallel(basnet)
            all_sal_masks = []
            for _, imgs, _ in tqdm(loader):
                sal_masks = basnet(imgs.cuda()).cpu()
                all_sal_masks.append(sal_masks)
            sal_masks = np.concatenate(all_sal_masks)
            np.savez(data_path, sal_masks=sal_masks)
        else:
            data = np.load(data_path)
            sal_masks = data["sal_masks"]
        return sal_masks

    def get_sal_mask(self, index):
        if self.sal.type == "dir":
            image_path = self.images[index]
            image_id = image_path.split("/")[-1][:-4]
            mask_path = os.path.join(self.sal.dir, f"{image_id}.png")
            mask = imageio.imread(mask_path)
            regions = regionprops(mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            # assert len(regions) == 1
            # we sort and take largest region for DeepUSPS
            region = regions[0]
            bbox = (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2])
            return mask.astype(np.int64), bbox
        return self.sal_masks[index].astype(np.int64), self.bbox[index]

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sal, bbox = self.get_sal_mask(index)
        assert sal.shape == image.shape[:2], f"Sal: {sal.shape}, Im: {image.shape}"
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


class PascalVOCSegmentationPseudolabels(PascalVOCSegmentation):
    def __init__(
        self,
        root,
        image_set,
        pseudolables_data_dir,
        transform,
        n_clusters=20,
    ):
        super().__init__(root=root, split=image_set, transform=transform)
        self.pseudolables_data_dir = pseudolables_data_dir
        pseudolabels_path = os.path.join(pseudolables_data_dir, "ids.npy")
        self.images = list(np.load(pseudolabels_path))
        self.n_clusters = n_clusters + 1

    def __getitem__(self, index):
        img_id = self.images[index]
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_file = os.path.join(self.pseudolables_data_dir, f"{img_id.split('/')[-1]}.npy")
        pseudo_mask = np.load(mask_file)
        assert pseudo_mask.shape == image.shape[:2]
        pseudo_mask = self._convert_to_segmentation_mask_cluster_lables(pseudo_mask)
        if self.transform is not None:
            transformed = self.transform(image=image, pseudo_mask=pseudo_mask)
            image = transformed["image"]
            pseudo_mask = transformed["pseudo_mask"].permute(2, 0, 1)

        return index, image, pseudo_mask

    def _convert_to_segmentation_mask_cluster_lables(self, mask):

        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, self.n_clusters), dtype=np.float32)
        for label_index in range(self.n_clusters):
            segmentation_mask[:, :, label_index] = (mask == label_index).astype(float)
        return segmentation_mask
