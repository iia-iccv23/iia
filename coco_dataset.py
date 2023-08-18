from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import skimage.io as io
import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import io


class Coco_Segmentation(Dataset):
    def __init__(
            self,
            root_path: Path,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transform_resize: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.annotations = COCO(f'{root_path}/annotations/instances_val2017.json')
        # self.imgs_path = Path(root_path / "images/val2017")
        self.imgs_path = Path(f'{root_path}/val2017')
        self.cat_ids = self.annotations.getCatIds()
        self.img_ids = self.get_img_ids()
        self.img_data = self.annotations.loadImgs(self.img_ids)
        self.files = [str(self.imgs_path / img["file_name"]) for img in self.img_data]
        self.transform = transform
        self.target_transform = target_transform
        self.transform_resize = transform_resize

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[index]['id'],
            catIds=self.cat_ids,
            iscrowd=None
        )
        anns = self.annotations.loadAnns(ann_ids)
        img = io.read_image(self.files[index])
        _img = T.ToPILImage()(img).convert('RGB')
        img_norm = self.get_image_by_index(_img)
        img_resize = self.transform_resize(_img)
        target = self.get_target_by_index(anns, img)

        return img_norm, target, img_resize

    def get_img_ids(self):
        valid_img_ids = []
        for cat in self.cat_ids:
            valid_img_ids.extend(self.annotations.getImgIds(catIds=cat))
        valid_img_ids = list(set(valid_img_ids))
        return valid_img_ids

    def get_image_by_index(self, img):
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_target_by_index(self, anns, img):
        nc, width, height = img.shape
        target = np.zeros((width, height))
        for val in anns:
            target = np.maximum(self.annotations.annToMask(val), target)
        target = Image.fromarray(target)
        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')  # onlyresize
            target = torch.from_numpy(target).long()
        return target

    def __len__(self) -> int:
        return len(self.files)