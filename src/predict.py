from src.model import UNET
from src.metrics import calculate_metrics
from scripts.data_visualize import save_comparison
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from pathlib import Path

def get_transforms():
    return A.Compose([
    A.Resize(256,256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict_single(model, image_dir, mask_dir=None, transforms=get_transforms(), output_dir="output/predictions", comparison_dir="output/comparison"):
    """Returns IoU and Dice scores of prediction, creates and saves mask and comparison images to provided directory"""

    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_dir)
    image_array = np.array(image, dtype=np.float32)

    min_hu = -1000
    max_hu = 400

    image_windowed = np.clip(image_array, min_hu, max_hu)

    image = ((image_windowed-min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)

    iou, dice = None, None
    y_true = None
    if mask_dir is not None:
        mask = Image.open(mask_dir)
        mask = (np.array(mask) > 0).astype(np.float32)

        transformed = transforms(image=image, mask=mask)
        y_true = transformed["mask"]
    else:
        transformed = transforms(image=image)

    image_tensor=transformed["image"]
    image_tensor=image_tensor.unsqueeze(0)

    model.eval()

    with torch.inference_mode():

        y_preds=model(image_tensor)

        if y_true is not None:
            iou, dice = calculate_metrics(y_preds, y_true)

        probs=torch.sigmoid(y_preds)
        pred_mask=(probs>0.5).float()
        mask_np=(pred_mask.squeeze().cpu().numpy()*255).astype(np.uint8)

    path=Path(image_dir)
    save_path=output_dir / f"{path.stem}.png"

    (Image.fromarray(mask_np)).save(save_path)

    print(f"Prediction saved: {save_path.name}")

    if y_true is not None:
        comparison_save_path = Path(comparison_dir) / f"comparison_{path.stem}.png"
        save_comparison(Image.fromarray(image), Image.open(mask_dir), Image.fromarray(mask_np), save_path=comparison_save_path)

    return iou, dice

def predict_multiple(model, image_folder_dir, mask_folder_dir=None, transforms=get_transforms(), output_dir="output/predictions"):
    image_folder = Path(image_folder_dir)

    iou_score, dice_score, count = 0, 0, 0

    for image_path in image_folder.iterdir():
        mask_path=None
        if mask_folder_dir is not None:
            mask_path = Path(mask_folder_dir) / image_path.name

            iou, dice = predict_single(model=model, image_dir=image_path, mask_dir=mask_path, transforms=transforms, output_dir=output_dir)

            if iou is not None and dice is not None:
                iou_score+=iou
                dice_score+=dice
                count+=1
        else:
            predict_single(model=model, image_dir=image_path, mask_dir=mask_path, transforms=transforms, output_dir=output_dir)

    if count > 0:
        print(f"Results of {count} images")
        print(f"mIoU : {iou_score / count:.4f}")
        print(f"mDICE : {dice_score / count:.4f}")


if __name__=="__main__":

    transforms=get_transforms()

    model=UNET()

    model.load_state_dict(torch.load("weights/100epochs.pt", map_location="cpu"))

    #iou, dice=predict_single(model=model, image_dir="data/split/test/images/ID_0001_Z_0146.tif", mask_dir="data/split/test/masks/ID_0001_Z_0146.tif", transforms=transforms)

    #print(iou, dice)
    predict_multiple(model=model, image_folder_dir="data/split/test/images", mask_folder_dir="data/split/test/masks", transforms=transforms)