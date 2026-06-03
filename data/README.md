# Local Data Directory

This directory is tracked only as a placeholder. Private and generated datasets are
ignored by Git.

Expected local files for the current workflow:

```text
data/raw_weld_256.npz          greyscale 256px dataset prepared from raw/*.jpg
data/raw_weld_256_montage.png  optional private input contact sheet
```

Recreate the main dataset with:

```bash
python prepare_dataset.py --input 'raw/*.jpg' --out data/raw_weld_256.npz --size 256
```
