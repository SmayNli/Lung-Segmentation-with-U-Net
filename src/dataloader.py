from torch.utils.data import DataLoader
from src.data_preperation import segmentation_transforms, SegmentationData

def create_dataloaders(train_image_dir="data/split/train/images",
                       train_mask_dir="data/split/train/masks",
                       val_image_dir="data/split/val/images",
                       val_mask_dir="data/split/val/masks",
                       batch_size=8):

    train_data=SegmentationData(train_image_dir, train_mask_dir, transform=segmentation_transforms)
    val_data=SegmentationData(val_image_dir, val_mask_dir, transform=segmentation_transforms)

    train_dataloader=DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader=DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader

if __name__=="__main__":
    train_dataloader, val_dataloader = create_dataloaders()

    print(train_dataloader, val_dataloader)

    X, y = next(iter(val_dataloader))

    print(len(y))
