# Lung CT Image Segmentation with U-Net

PyTorch implementation of the classic U-Net architecture built from scratch for binary lung segmentation on 32-bit thoracic Computerized Tomography (CT) scans.

---

## Highlights & Results
* **Architecture:** Pure PyTorch implementation of the U-Net contracting-expansive path with skip connections.
* **Medical Domain Knowledge:** Solved the contrast degradation issue on 32-bit signed Hounsfield Unit (HU) CT data by applying **Lung Windowing (`[-1000, 400] HU`)** instead of naive 8-bit clipping.
* **Custom Loss Function:** Combined Binary Cross Entropy with Dice Loss (`DiceBCELoss`).
* **Test Performance (41 Unseen CT Slices):**
  * **Mean IoU (Jaccard Index):** `94.70%`
  * **Mean Dice Coefficient (F1 Score):** `96.71%`

---

## Qualitative Results

Here is a side-by-side comparisons showing the Input CT, Ground Truth Mask, Model Prediction, and Colored Overlap:

![Comparison](assets/sample_comparison1.png)
![Comparison](assets/sample_comparison2.png)
---

## Project Structure

```text
├── data/                  # CT scans and masks (ignored in git)
├── output/
│   ├── predictions/       # Saved binary mask predictions
│   └── comparison/        # 4-panel visual comparison figures
├── scripts/
│   ├── data_split.py      # Deterministic train/val/test splitter
│   └── data_visualize.py  # Matplotlib 4-panel overlap visualizer
├── src/
│   ├── model.py           # Vanilla U-Net architecture from scratch
│   ├── data_preperation.py# HU Windowing, Albumentations transforms & Dataset
│   ├── dataloader.py      # PyTorch DataLoader wrappers
│   ├── loss_function.py   # Custom DiceBCELoss
│   ├── metrics.py         # IoU & Dice calculation functions
│   ├── train.py           # Training & validation loop
│   └── predict.py         # Single & batch inference pipeline
├── weights/               # Trained checkpoints (100epochs.pt)
└── requirements.txt       # Dependencies
```

## How to Run

### 1. Data Split
Split your paired CT images and masks into 70% Train, 15% Validation, and 15% Test sets:
```bash
python scripts/data_split.py
```

### 2. Training
Train the U-Net model from scratch using custom `DiceBCELoss` and evaluation metrics:
```bash
python -m src.train
```

### 3. Inference & Visual Evaluation
Run inference on the test set, calculate **mIoU & mDICE**, and generate 4-panel comparison images:
```bash
python -m src.predict
```
*Generated predictions will be saved to `output/predictions/` and 4-panel overlays to `output/comparison/`.*

---

## Model Weights

Pre-trained weights (`100epochs.pt`) are available under the **[GitHub Releases](https://github.com/SmayNli/Lung-Segmentation-with-U-Net/releases/tag/v1.0.0)** page. 

You can download and place them directly into the `weights/` directory using your terminal:

### Windows (PowerShell):
```powershell
New-Item -ItemType Directory -Force -Path weights
Invoke-WebRequest -Uri "https://github.com/SmayNli/Lung-Segmentation-with-U-Net/releases/download/v1.0.0/100epochs.pt" -OutFile "weights/100epochs.pt"
```

### Linux / macOS:
```bash
mkdir -p weights
curl -L -o weights/100epochs.pt "https://github.com/SmayNli/Lung-Segmentation-with-U-Net/releases/download/v1.0.0/100epochs.pt"
```