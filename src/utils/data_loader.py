import os
import json
import pandas as pd

# Default tags we care about

def load_dataset_split(split_dir):
    """
    Load all .json files in a given split directory (train/val/test).
    Combines problem description + code and filters by target tags.
    
    Args:
        split_dir (str): Path to directory containing .json samples.

    Returns:
        pd.DataFrame with columns ['text', 'tags']
    """
    samples = []

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f" XXX ==== Directory not found: {split_dir} ====")

    for fname in os.listdir(split_dir):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(split_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                sample = json.load(f)
        except json.JSONDecodeError:
            print(f" /!\ ==== Skipping invalid JSON file: {fpath} ====")
            continue

        desc = sample.get("prob_desc_description", "") or ""
        code = sample.get("source_code", "") or ""
        tags = sample.get("tags", [])

        if tags:
            samples.append({
                "text": desc.strip() + "\n\n" + code.strip(),
                "tags": tags
            })

    df = pd.DataFrame(samples)
    print(f"==== Loaded {len(df)} samples from {split_dir} ====")
    return df


def load_all_splits(data_root):
    """
    Load train/val/test splits under a common data root.

    Args:
        data_root (str): Root directory containing 'train', 'val', 'test' folders.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_df = load_dataset_split(train_dir)
    val_df = load_dataset_split(val_dir)
    test_df = load_dataset_split(test_dir)

    return train_df, val_df, test_df
