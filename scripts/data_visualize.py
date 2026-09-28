import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image

def save_comparison(image, true_mask, pred_mask, save_path):
    """Creates comparison image by using provided image, true_mask and pred_mask. In last image it overlays image and pred_mask"""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = image.resize((256, 256))
    true_mask = true_mask.resize((256, 256))
    pred_mask = pred_mask.resize((256, 256))

    pred_mask = np.array(pred_mask)
    masked_overlay = np.ma.masked_where(pred_mask == 0, pred_mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")
    
    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(masked_overlay, cmap="Reds", alpha=0.50)
    axes[3].set_title("Overlay")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

if __name__ == "__main__":

    tensor1=np.array(np.random.randn(256,256))
    tensor2=np.array(np.random.randn(256,256))
    tensor3=np.array(np.random.randn(256,256))

    tensor1=Image.fromarray(tensor1)
    tensor2=Image.fromarray(tensor2)
    tensor3=Image.fromarray(tensor3)

    save_comparison(tensor1, tensor2, tensor3, save_path="")