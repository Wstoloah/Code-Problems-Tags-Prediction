import os
import json
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

def load_samples(src_dir, target_tags):
    """
    Load all JSON files from src_dir.
    Keep all samples, but identify which tags overlap with TARGET_TAGS.
    """
    data = []
    files = [f for f in os.listdir(src_dir) if f.endswith(".json")]
    
    for fn in tqdm(files, desc="Loading JSON files"):
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

            # Always keep the sample
            filtered_tags = [t for t in tags if t in target_tags]
            data.append((fn, tags, filtered_tags))

        except Exception as e:
            print(f"[WARN] skipping {fn}: {e}")

    return data


def stratified_split_multilabel(files, filtered_tags_list, test_size=0.2, val_size=0.1, random_state=42):
    """
    Perform a stratified split on multi-label data using only TARGET_TAGS for balancing.
    """
    mlb = MultiLabelBinarizer(classes=TARGET_TAGS)
    Y = mlb.fit_transform(filtered_tags_list)
    indices = np.arange(len(files))

    # Primary split (train/test)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=(Y.sum(axis=1) > 0)  # simple stratification by having target tags or not
    )

    # Validation split (from training portion)
    train_files, val_files, train_tags, val_tags = train_test_split(
        [files[i] for i in train_idx],
        [filtered_tags_list[i] for i in train_idx],
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    test_files = [files[i] for i in test_idx]
    return train_files, val_files, test_files


def copy_and_save_json(files, all_tags_map, filtered_tags_map, dest_dir):
    """
    Copy selected JSON files into destination directory.
    Each output JSON includes:
      - original tags
      - filtered target tags
    """
    os.makedirs(dest_dir, exist_ok=True)
    for fn in tqdm(files, desc=f"Copying to {os.path.basename(dest_dir)}"):
        src_path = os.path.join(SRC_DIR, fn)
        dst_path = os.path.join(dest_dir, fn)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            obj["original_tags"] = all_tags_map[fn]
            obj["filtered_tags"] = filtered_tags_map[fn]

            with open(dst_path, "w", encoding="utf-8") as out:
                json.dump(obj, out, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[WARN] Could not copy {fn}: {e}")


def main():
    # --- Load all samples ---
    samples = load_samples(SRC_DIR, TARGET_TAGS)
    if not samples:
        print("No valid samples found.")
        return

    files, all_tags, filtered_tags = zip(*samples)
    filtered_map = {f: ft for f, ft in zip(files, filtered_tags)}
    all_map = {f: at for f, at in zip(files, all_tags)}

    print(f"\nTotal samples loaded: {len(samples)}")
    num_with_target = sum(1 for t in filtered_tags if len(t) > 0)
    print(f"Samples containing at least one target tag: {num_with_target} ({num_with_target/len(samples)*100:.1f}%)")

    # --- Split dataset ---
    train_files, val_files, test_files = stratified_split_multilabel(files, filtered_tags)
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")

    # --- Save data ---
    copy_and_save_json(train_files, all_map, filtered_map, TRAIN_DIR)
    copy_and_save_json(val_files, all_map, filtered_map, VAL_DIR)
    copy_and_save_json(test_files, all_map, filtered_map, TEST_DIR)

    print("\n==== Split completed and saved under: ====")
    print(f"  - {TRAIN_DIR}\n  - {VAL_DIR}\n  - {TEST_DIR}")


if __name__ == "__main__":
    main()
