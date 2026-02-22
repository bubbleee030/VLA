import h5py
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# HDF5 dataset 路徑
dataset_dirs = [
    "data/datasets/mango_new_samples_span_hdf5_gelsight",
    "data/datasets/wipe_samples_span_hdf5_gelsight"
]

dataset_stats = {}

for dataset_dir in dataset_dirs:
    dataset_dir_path = Path(dataset_dir)
    if not dataset_dir_path.exists():
        print(f"Dataset folder not found: {dataset_dir}")
        continue

    dataset_name = dataset_dir_path.name
    print(f"Processing dataset: {dataset_name}")

    all_states = []

    for h5_file in tqdm(sorted(dataset_dir_path.glob("*.h5"))):
        with h5py.File(h5_file, "r") as f:
            if "states" in f:
                states = np.array(f["states"])
                all_states.append(states)
            else:
                print(f"  Warning: 'states' not found in {h5_file.name}")

    if all_states:
        all_states = np.concatenate(all_states, axis=0)
        stats = {
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
            "num_episodes": len(dataset_dir_path.glob("*.h5")),
            "num_samples": all_states.shape[0]
        }
        dataset_stats[dataset_name] = stats
    else:
        print(f"  No 'states' data found in {dataset_name}, skipping...")

# 儲存統計結果
output_path = Path("configs/dataset_stat.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(dataset_stats, f, indent=2)

print(f"Dataset stats saved to {output_path}")

