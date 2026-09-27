import random
import shutil
from pathlib import Path

# Paths configuration
# Resolve paths relative to project root so the script works from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "2d_images"
MASKS_DIR = DATA_DIR / "2d_masks"
OUTPUT_DIR = DATA_DIR / "split"

# Split ratios: Train 70%, Val 15%, Test 15%
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def split_data():
    # Verify input directories exist
    if not IMAGES_DIR.exists() or not MASKS_DIR.exists():
        raise FileNotFoundError(
            f"Images or masks directory not found!\n"
            f"Expected images at: {IMAGES_DIR}\n"
            f"Expected masks at: {MASKS_DIR}"
        )

    # Map files by stem (filename without extension) to match images and masks safely
    image_map = {f.stem: f for f in IMAGES_DIR.iterdir() if f.is_file()}
    mask_map = {f.stem: f for f in MASKS_DIR.iterdir() if f.is_file()}

    # Find paired samples
    common_keys = sorted(list(set(image_map.keys()) & set(mask_map.keys())))
    total_samples = len(common_keys)

    if total_samples == 0:
        print("No matching image/mask pairs found.")
        return

    # Check for any unmatched files
    missing_masks = set(image_map.keys()) - set(mask_map.keys())
    missing_images = set(mask_map.keys()) - set(image_map.keys())
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images without matching masks.")
    if missing_images:
        print(f"Warning: {len(missing_images)} masks without matching images.")

    # Shuffle deterministically using standard library
    random.seed(RANDOM_SEED)
    shuffled_keys = common_keys.copy()
    random.shuffle(shuffled_keys)

    # Calculate split index boundaries
    train_end = int(total_samples * TRAIN_RATIO)
    val_end = train_end + int(total_samples * VAL_RATIO)

    splits = {
        "train": shuffled_keys[:train_end],
        "val": shuffled_keys[train_end:val_end],
        "test": shuffled_keys[val_end:],
    }

    print(f"Found {total_samples} paired samples.")
    print(
        f"Splitting into -> Train: {len(splits['train'])}, "
        f"Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )

    # Copy files into organized split directories
    for split_name, keys in splits.items():
        split_img_dir = OUTPUT_DIR / split_name / "images"
        split_mask_dir = OUTPUT_DIR / split_name / "masks"

        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        for k in keys:
            img_file = image_map[k]
            mask_file = mask_map[k]
            shutil.copy2(img_file, split_img_dir / img_file.name)
            shutil.copy2(mask_file, split_mask_dir / mask_file.name)

    print(f"Done! Split dataset successfully saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    split_data()