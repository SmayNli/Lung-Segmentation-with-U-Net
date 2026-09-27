import os
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

segmentation_transforms=A.Compose([
    A.Resize(256,256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class SegmentationData(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir

        self.images=os.listdir(image_dir)
        self.masks=os.listdir(mask_dir)
        self.transform=transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path=os.path.join(self.image_dir, self.images[idx])
        mask_path=os.path.join(self.mask_dir, self.masks[idx])

        image=Image.open(image_path)

        image_array = np.array(image, dtype = np.float32)

        min_hu = -1000
        max_hu = 400

        image_windowed=np.clip(image_array, min_hu, max_hu)
        image = ((image_windowed - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)

        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        mask = (mask_array > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image=augmented["image"]
            mask=augmented["mask"]

        mask=torch.unsqueeze(mask, 0)

        return image, mask

if __name__=="__main__":

    data=SegmentationData("data/split/test/images", "data/split/test/masks", segmentation_transforms)

    image, mask=data[5]
    print(mask.max(), mask.min())