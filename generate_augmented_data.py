#!/usr/bin/env python3
"""
資料增量與視覺化腳本 (終極修正版)
- 徹底修正 GaussianBlur 差異圖問題
"""

import h5py
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os

# 確保中文字體顯示正常
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AddGaussianNoise:
    def __init__(self, mean=0., std=10.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        tensor_255 = tensor * 255.0
        noisy_tensor_255 = tensor_255 + torch.randn(tensor.size()) * self.std + self.mean
        return noisy_tensor_255.clamp(0, 255) / 255.0
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def augment_and_visualize(h5_file_path, output_dir='./plots'):
    if not os.path.exists(h5_file_path):
        print(f"❌ 錯誤：找不到 HDF5 檔案 {h5_file_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file_path, 'r') as f:
        original_image_np = f['gelsight/gelsight'][4]
    
    original_tensor = torch.from_numpy(original_image_np).permute(2, 0, 1)

    # 💡 增加模糊強度，確保效果可見
    augmentations = {
        '輕微顏色變化': T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        '模擬感測器模糊': T.GaussianBlur(kernel_size=9, sigma=2.0),
        '模擬電子雜訊': AddGaussianNoise(std=8.0),
        '模擬彈性形變': T.ElasticTransform(alpha=35.0, sigma=4.0),
    }

    fig, axes = plt.subplots(2, len(augmentations) + 1, figsize=(20, 8))
    fig.suptitle('GelSight 觸覺資料增量', fontsize=20, fontweight='bold')

    axes[0, 0].imshow(original_image_np)
    axes[0, 0].set_title('原始影像', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_image_np)
    axes[1, 0].set_title(' ', fontsize=12)
    axes[1, 0].axis('off')

    for i, (name, aug) in enumerate(augmentations.items(), 1):
        
        # --- 💡 終極修正邏輯 ---
        # 統一將原圖轉為 float32 [0, 1] 範圍的高精度張量
        original_float_tensor = original_tensor.to(torch.float32) / 255.0
        
        # 應用增量
        if isinstance(aug, T.GaussianBlur):
            # 對於模糊，我們在 float 張量上操作
            augmented_float_tensor = aug(original_float_tensor)
        elif isinstance(aug, AddGaussianNoise):
             # 雜訊函式內部處理了範圍轉換
            augmented_float_tensor = aug(original_float_tensor)
        else: # ColorJitter, ElasticTransform
            # 其他函式也都在 float 張量上操作
            augmented_float_tensor = aug(original_float_tensor)

        # 將增量後的影像轉回 uint8 [0, 255] 以便顯示
        augmented_image_np = (augmented_float_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        
        # --- 差異計算與增強 (使用高精度 float32 計算) ---
        # 將原圖也轉為 float32 [0, 255]
        original_image_float = original_image_np.astype(np.float32)
        # 將增量圖也轉為 float32 [0, 255]
        augmented_image_float = augmented_image_np.astype(np.float32)

        diff_float = np.abs(original_image_float - augmented_image_float)
        
        max_diff = diff_float.max()
        if max_diff > 1e-5:
            diff_enhanced = (diff_float / max_diff * 255.0).astype(np.uint8)
        else:
            diff_enhanced = diff_float.astype(np.uint8)

        # --- 繪圖 ---
        axes[0, i].imshow(augmented_image_np)
        axes[0, i].set_title(name, fontsize=14)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(diff_enhanced)
        axes[1, i].set_title('與原圖的差異 (增強顯示)', fontsize=12, color='red')
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, 'gelsight_augmentation_ultimate_fix.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 終極修正後的對比圖已儲存到：{save_path}")
    plt.show()

if __name__ == "__main__":
    h5_file = './data/datasets/mango_hdf5_gelsight/episode_0.h5'
    augment_and_visualize(h5_file)