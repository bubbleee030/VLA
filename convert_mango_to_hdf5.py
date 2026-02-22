import os
import h5py
import numpy as np
import json
from pathlib import Path
import cv2
import re
import torch

def get_file_number(filename):
    # Extract the number between "rgb_" and ".jpg"
    match = re.search(r'rgb_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return 0

def convert_episode_to_hdf5(episode_path, output_path):
    """
    Convert a single episode's data to HDF5 format, handling both images and numpy files
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving HDF5 file to: {output_path}")
    with h5py.File(output_path, 'w') as hf:
        for item_name in os.listdir(episode_path):
            item_path = os.path.join(episode_path, item_name)

            # Handle numpy files directly
            if os.path.isfile(item_path) and item_path.endswith('.npy'):
                data = np.load(item_path, allow_pickle=True)
                base_name = os.path.splitext(item_name)[0]
                hf.create_dataset(base_name, data=data, compression='lzf')
                print(f"  Saved numpy file: {item_name}, size {data.shape}")
                continue

            # Handle .pt files
            elif os.path.isfile(item_path) and item_path.endswith('.pt'):
                embedding_tensor = torch.load(item_path)
                embedding_np = embedding_tensor.to(torch.float32).numpy()
                hf.create_dataset("instruct_embeddings", data=embedding_np, compression='lzf')
                print(f"  Saved pt file: {item_name}, size {embedding_tensor.shape}")
                continue

            if not os.path.isdir(item_path):
                continue

            sensor_group = hf.create_group(item_name)
            files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files.sort(key=get_file_number)

            if image_files:
                all_images = []
                try:
                    for file_name in image_files:
                        file_path = os.path.join(item_path, file_name)
                        img = cv2.imread(file_path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            all_images.append(img_rgb)

                    if all_images:
                        images_array = np.array(all_images)
                        sensor_group.create_dataset(item_name, data=images_array, compression='lzf')
                        print(f"  Saved {len(all_images)} images from {item_name}, size {images_array.shape}")

                except ValueError:
                    print(f"  Images in {item_name} have different dimensions. Resizing...")
                    min_height = float('inf')
                    min_width = float('inf')
                    for file_name in image_files:
                        file_path = os.path.join(item_path, file_name)
                        img = cv2.imread(file_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            min_height = min(min_height, h)
                            min_width = min(min_width, w)

                    resized_images = []
                    for file_name in image_files:
                        file_path = os.path.join(item_path, file_name)
                        img = cv2.imread(file_path)
                        if img is not None:
                            resized = cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_AREA)
                            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            resized_images.append(resized_rgb)

                    if resized_images:
                        resized_array = np.array(resized_images)
                        sensor_group.create_dataset(item_name, data=resized_array, compression='lzf')
                        sensor_group.attrs['resized'] = True
                        sensor_group.attrs['target_width'] = min_width
                        sensor_group.attrs['target_height'] = min_height
                        print(f"  Saved {len(resized_images)} resized images ({min_width}x{min_height}) from {item_name}")

            npy_files = [f for f in files if f.endswith('.npy')]
            if npy_files:
                npy_files.sort(key=get_file_number)
                for file_name in npy_files:
                    file_path = os.path.join(item_path, file_name)
                    data = np.load(file_path)
                    base_name = os.path.splitext(file_name)[0]
                    sensor_group.create_dataset(base_name, data=data, compression='lzf')
                print(f"  Saved {len(npy_files)} numpy files from {item_name}")

def convert_dataset_to_hdf5(input_dir, output_dir):
    """
    Convert all episodes in the dataset to HDF5 format
    """
    os.makedirs(output_dir, exist_ok=True)
    episode_dirs = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # 修正排序，抓最後一段數字
    def extract_number(name):
        parts = re.findall(r'\d+', name)
        return int(parts[-1]) if parts else 0

    episode_dirs.sort(key=extract_number)

    for episode_dir in episode_dirs:
        input_path = os.path.join(input_dir, episode_dir)
        output_path = os.path.join(output_dir, f"{episode_dir}.h5")
        print(f"Converting {episode_dir} to HDF5...")
        convert_episode_to_hdf5(input_path, output_path)
        print(f"Converted {episode_dir} successfully")

if __name__ == "__main__":
    # 🔧 修改：指定您的輸入和輸出路徑
    input_directory = "/home/cmwang16/VLA/data/datasets/mango"
    output_directory = "/home/cmwang16/VLA/data/datasets/mango_hdf5_gelsight"
    
    print(f"📂 輸入目錄: {input_directory}")
    print(f"📁 輸出目錄: {output_directory}")
    print("="*60)
    
    # 檢查輸入目錄是否存在
    if not os.path.exists(input_directory):
        print(f"❌ 錯誤：找不到輸入目錄 {input_directory}")
        exit(1)
    
    # 列出所有 episode
    episodes = [f for f in os.listdir(input_directory) 
                if os.path.isdir(os.path.join(input_directory, f)) and f.startswith('episode_')]
    
    print(f"✅ 找到 {len(episodes)} 個 episodes")
    print(f"   範例: {episodes[:3]}")
    print("="*60)
    
    # 轉換
    convert_dataset_to_hdf5(input_directory, output_directory)
    
    print("\n" + "="*60)
    print("✅ 轉換完成！")
    print(f"📁 HDF5 檔案位置: {output_directory}")
    print("="*60)
