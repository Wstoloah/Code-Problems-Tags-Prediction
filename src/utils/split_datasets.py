import os
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# --- CONFIG ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

SRC_DIR = os.path.join(PROJECT_ROOT, "data", "code_classification_dataset")
OUT_BASE = os.path.join(PROJECT_ROOT, "data")

TRAIN_DIR = os.path.join(OUT_BASE, "train")
VAL_DIR   = os.path.join(OUT_BASE, "val")
TEST_DIR  = os.path.join(OUT_BASE, "test")

TARGET_TAGS = [
    'math', 'graphs', 'strings', 'number theory',
    'trees', 'geometry', 'games', 'probabilities'
]
# --- END CONFIG ---

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def load_and_filter_samples(src_dir, target_tags):
    data = []
    files = [f for f in os.listdir(src_dir) if f.endswith(".json")]
    for fn in tqdm(files, desc="Loading & filtering JSON files"):
        path = os.path.join(src_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            tags = obj.get("tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [tags]
            # Filter tags to only relevant ones
            filtered_tags = [t for t in tags if t in target_tags]
            if not filtered_tags:
                continue  # skip samples with no relevant tag
            data.append((fn, filtered_tags))
        except Exception as e:
            print(f"[WARN] skipping {fn}: {e}")
    return data

def stratified_split_multilabel(files, tags_list, test_size=0.2, val_size=0.1, random_state=42):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags_list)
    indices = np.arange(len(files))

    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    # Second split: validation from train
    train_files, val_files, train_tags, val_tags = train_test_split(
        [files[i] for i in train_idx],
        [tags_list[i] for i in train_idx],
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )
    test_files = [files[i] for i in test_idx]
    return train_files, val_files, test_files

def copy_and_trim_tags(files, tags_map, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fn in tqdm(files, desc=f"Copying to {dest_dir}"):
        src_path = os.path.join(SRC_DIR, fn)
        dst_path = os.path.join(dest_dir, fn)
        with open(src_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        obj["tags"] = tags_map[fn]  # overwrite with filtered tags
        with open(dst_path, "w", encoding="utf-8") as out:
            json.dump(obj, out, ensure_ascii=False, indent=2)

def main():
    samples = load_and_filter_samples(SRC_DIR, TARGET_TAGS)
    if not samples:
        print("No matching JSON files found with target tags.")
        return
    files, tags_list = zip(*samples)
    tags_map = {f: t for f, t in samples}

    train_files, val_files, test_files = stratified_split_multilabel(files, tags_list)
    print(f"\nKept {len(samples)} samples total.")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    copy_and_trim_tags(train_files, tags_map, TRAIN_DIR)
    copy_and_trim_tags(val_files, tags_map, VAL_DIR)
    copy_and_trim_tags(test_files, tags_map, TEST_DIR)
    print("\n==== Split completed and saved under: ====")
    print(f"  - {TRAIN_DIR}\n  - {VAL_DIR}\n  - {TEST_DIR}")

if __name__ == "__main__":
    main()