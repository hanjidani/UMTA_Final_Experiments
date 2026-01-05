# Dataset Setup Guide

## ImageNet-100 Dataset Structure

The code expects ImageNet-100 to be organized as follows:

```
data/
└── imagenet-100/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── class2/
    │   └── ...
    └── val/
        ├── class1/
        ├── class2/
        └── ...
```

## Downloading ImageNet-100

### Option 1: Download Pre-made ImageNet-100

1. Download from official sources or research datasets
2. Extract to `data/imagenet-100/`
3. Ensure structure matches above

### Option 2: Create from Full ImageNet

If you have full ImageNet:

```python
# Script to create ImageNet-100 subset
# Select 100 classes and copy images
```

### Option 3: Use Kaggle Dataset

Many Kaggle datasets provide ImageNet-100:
- Search for "ImageNet-100" or "ImageNet subset"
- Download and extract to `data/imagenet-100/`

## For Kaggle Notebooks

If running on Kaggle, you can:

1. **Add ImageNet-100 as a Kaggle dataset:**
   - Upload ImageNet-100 as a Kaggle dataset
   - Add it to your notebook
   - Update path in code if needed

2. **Download in notebook:**
   ```python
   # In Kaggle notebook
   !wget [URL_TO_IMAGENET100] -O imagenet100.zip
   !unzip imagenet100.zip -d /kaggle/working/data/
   ```

## Verification

After setup, verify structure:

```python
from pathlib import Path
data_dir = Path("data/imagenet-100")
print(f"Train classes: {len(list((data_dir / 'train').iterdir()))}")
print(f"Val classes: {len(list((data_dir / 'val').iterdir()))}")
```

Expected: 100 classes in both train and val directories.

