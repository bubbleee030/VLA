#!/usr/bin/env python3
"""
資料增量腳本：
讀取原始 HDF5 資料集，對觸覺影像進行隨機增量，並生成一個更大的新資料集。
"""

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
import os

def get_augmentation_pipeline():
    """定義我們的隨機增量流程"""
    # 這些增量會以一定的機率被隨機應用
    return T.Compose([
        T.RandomApply([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        ], p=0.8),
        T.RandomApply([
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
        ], p=0.5),
        T.RandomApply([
            T.ElasticTransform(alpha=35.0, sigma=4.0)
        ], p=0.5),
    ])

def process_episode(input_file, output_path, aug_pipeline):
    """處理單一 episode 檔案，應用增量並儲存"""
    with h5py.File(input_file, 'r') as hf_in:
        with h5py.File(output_path, 'w') as hf_out:
            # 1. 複製所有非影像資料 (例如機器人狀態)
            for key, item in hf_in.items():
                if not isinstance(item, h5py.Group):
                    hf_out.create_dataset(key, data=item[()])
            
            # 2. 對觸覺影像進行增量
            if 'gelsight/gelsight' in hf_in:
                original_images_np = hf_in['gelsight/gelsight'][()]
                
                augmented_images = []
                for img_np in original_images_np:
                    # 將影像轉為 tensor [C, H, W]
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                    
                    # 應用增量 (只在 uint8 上操作)
                    augmented_tensor = aug_pipeline(img_tensor)
                    
                    # 轉回 numpy [H, W, C]
                    augmented_np = augmented_tensor.permute(1, 2, 0).numpy()
                    augmented_images.append(augmented_np)
                
                # 儲存增量後的影像
                hf_out.create_dataset('gelsight/gelsight', data=np.array(augmented_images))
            
            # 3. (可選) 您也可以對其他相機影像做同樣的增量
            for camera in ['camera1', 'camera2']:
                dset_name = f'{camera}/{camera}'
                if dset_name in hf_in:
                     hf_out.create_dataset(dset_name, data=hf_in[dset_name][()])


def augment_dataset(input_dir, output_dir, augmentation_factor=5):
    """
    對整個資料集進行增量

    Args:
        input_dir (str): 原始 HDF5 資料夾路徑
        output_dir (str): 增量後資料儲存的路徑
        augmentation_factor (int): 增量倍數，每個原始檔案會生成 N 個增量版本
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 取得原始檔案列表
    original_files = sorted(list(input_path.glob('*.h5')))
    if not original_files:
        print(f"❌ 錯誤：在 '{input_dir}' 中找不到任何 .h5 檔案")
        return
        
    print(f"📂 找到 {len(original_files)} 個原始 episode 檔案。")
    print(f"⚙️ 增量倍數：{augmentation_factor}x")
    print(f"💾 輸出目錄：{output_dir}")
    print("-" * 60)

    # 取得增量流程
    aug_pipeline = get_augmentation_pipeline()

    # 使用 tqdm 顯示進度條
    with tqdm(total=len(original_files) * augmentation_factor, desc="資料增量中") as pbar:
        # 1. 先將原始檔案複製一份到新目錄
        for original_file in original_files:
            new_path = output_path / original_file.name
            if not new_path.exists():
                os.link(original_file, new_path) # 使用硬連結，快速且不佔空間
            pbar.set_postfix_str(f"複製: {original_file.name}")
        
        # 2. 生成增量檔案
        for i in range(augmentation_factor - 1): # -1 因為原始檔算一份
            for original_file in original_files:
                output_filename = f"{original_file.stem}_aug_{i+1}.h5"
                output_filepath = output_path / output_filename
                
                process_episode(original_file, output_filepath, aug_pipeline)
                pbar.update(1)
                pbar.set_postfix_str(f"生成: {output_filename}")
    
    total_files = len(list(output_path.glob('*.h5')))
    print("-" * 60)
    print(f"✅ 資料增量完成！")
    print(f"   總共生成了 {total_files} 個 HDF5 檔案。")

if __name__ == "__main__":
    # --- 參數設定 ---
    INPUT_DATA_DIR = './data/datasets/mango_hdf5_gelsight'
    OUTPUT_DATA_DIR = './data/datasets/mango_hdf5_augmented_30x'
    AUGMENTATION_FACTOR = 30  # 我們將資料擴充到 30 倍
    
    augment_dataset(INPUT_DATA_DIR, OUTPUT_DATA_DIR, AUGMENTATION_FACTOR)