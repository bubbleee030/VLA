import h5py
import glob
import os

data_dir = "data/datasets/mango_new_samples_span_hdf5_gelsight"
h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

for h5_file in h5_files:
    print(f"\n=== {os.path.basename(h5_file)} ===")
    try:
        with h5py.File(h5_file, "r") as f:
            def print_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  [Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"  [Group]   {name}")

            f.visititems(print_keys)
    except Exception as e:
        print(f"  Error reading file: {e}")
