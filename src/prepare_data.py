# prepare_data.py
import os, shutil, random
from pathlib import Path
from PIL import Image

DATA_DIR = Path("dataset")   # original dataset with class folders
OUT_DIR = Path("data_split") # output split
SIZES = (64,64)              # image size

for cls in DATA_DIR.iterdir():
    if not cls.is_dir(): continue
    images = list(cls.glob("*"))
    random.shuffle(images)
    n = len(images)
    n_train = int(0.7*n)
    n_val = int(0.15*n)
    train = images[:n_train]; val = images[n_train:n_train+n_val]; test = images[n_train+n_val:]

    for split, items in [("train", train), ("val", val), ("test", test)]:
        target = OUT_DIR / split / cls.name
        target.mkdir(parents=True, exist_ok=True)
        for img in items:
            # optional: resize and save copy
            try:
                im = Image.open(img).convert("RGB")
                im = im.resize(SIZES)
                im.save(target / img.name)
            except Exception as e:
                print("skip", img, e)
print("done")
