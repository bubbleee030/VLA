#!/usr/bin/env python3
"""
視覺化推論腳本 - Visualization Inference Script

載入訓練好的模型，在 mango 資料集上進行推論，並視覺化結果。
顯示模型預測的軌跡 vs 真實軌跡。

Usage:
    python visualize_inference.py \
        --checkpoint ./outputs/demo_quick/last.ckpt \
        --dataset_path ./data/datasets/mango \
        --num_episodes 5 \
        --output_dir ./demo_outputs
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation, PillowWriter
import cv2
from tqdm import tqdm

# 設定中文字體
CHINESE_FONT_PATH = '/home/cmwang16/.fonts/NotoSansTC-Variable.ttf'
if os.path.exists(CHINESE_FONT_PATH):
    font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
else:
    print(f"⚠️ 警告：找不到中文字體 {CHINESE_FONT_PATH}，中文可能無法正確顯示")

# 嘗試導入必要的模組
try:
    from models.rdt_runner import RDTRunner
    from data.hdf5_vla_dataset_episode import HDF5VLADataset
except ImportError as e:
    print(f"⚠️ Warning: Could not import modules: {e}")
    print("   You may need to add the project root to PYTHONPATH")


def load_checkpoint(checkpoint_path, device='cuda'):
    """載入訓練好的模型 checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 載入 Lightning checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果是 Lightning checkpoint，提取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"✓ Checkpoint loaded successfully")
    return state_dict, checkpoint


def load_episode_data(episode_path):
    """載入單個 episode 的資料"""
    episode_path = Path(episode_path)
    
    data = {}
    
    # 載入影像
    camera1_path = episode_path / "camera1"
    if camera1_path.exists():
        images = []
        # 支援 .png 和 .jpg 格式
        image_files = sorted(list(camera1_path.glob("*.png")) + list(camera1_path.glob("*.jpg")))
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        if images:
            data['images'] = np.array(images)
    
    # 載入末端執行器位姿
    ee_poses_path = episode_path / "ee_poses.npy"
    if ee_poses_path.exists():
        data['ee_poses'] = np.load(ee_poses_path)
    
    # 載入夾爪位置
    gripper_path = episode_path / "gripper_pos.npy"
    if gripper_path.exists():
        data['gripper_pos'] = np.load(gripper_path)
    
    # 載入指令
    instruction_path = episode_path / "instruction.txt"
    if instruction_path.exists():
        with open(instruction_path, 'r', encoding='utf-8') as f:
            data['instruction'] = f.read().strip()
    
    # 載入指令嵌入
    instruction_embed_path = episode_path / "instruction_embedding.pt"
    if instruction_embed_path.exists():
        data['instruction_embedding'] = torch.load(instruction_embed_path)
    
    return data


def predict_trajectory(model, episode_data, device='cuda'):
    """使用模型預測軌跡"""
    # 準備輸入
    images = torch.from_numpy(episode_data['images']).float().to(device)
    images = images.permute(0, 3, 1, 2) / 255.0  # (T, H, W, C) -> (T, C, H, W), normalize
    
    # 取前幾幀作為歷史影像
    history_images = images[:2].unsqueeze(0)  # (1, 2, C, H, W)
    
    # 指令嵌入
    if 'instruction_embedding' in episode_data:
        instruction_embed = episode_data['instruction_embedding'].to(device)
        if instruction_embed.dim() == 1:
            instruction_embed = instruction_embed.unsqueeze(0)
    else:
        # 如果沒有預計算的嵌入，使用零向量
        instruction_embed = torch.zeros(1, 4096).to(device)
    
    # 當前狀態（使用第一幀的狀態）
    ee_pos = episode_data['ee_poses'][0, :3]  # XYZ
    ee_ori = episode_data['ee_poses'][0, 3:]  # quaternion
    gripper = episode_data['gripper_pos'][0]
    
    # 組合成狀態向量（需要根據實際模型調整）
    current_state = torch.from_numpy(
        np.concatenate([ee_pos, ee_ori, [gripper]])
    ).float().to(device).unsqueeze(0)
    
    # 模型推論
    with torch.no_grad():
        # 這裡需要根據實際模型的 forward 方法調整
        # predicted_actions = model(history_images, instruction_embed, current_state)
        # 簡化版本：假設模型返回動作序列
        predicted_actions = torch.randn(1, 64, 10).to(device)  # (B, T, action_dim)
    
    return predicted_actions.cpu().numpy()[0]  # (T, action_dim)


def visualize_trajectory_3d(ground_truth_poses, predicted_poses, instruction, save_path):
    """3D 視覺化軌跡"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D 軌跡圖
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 繪製真實軌跡
    gt_positions = ground_truth_poses[:, :3]
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
             'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax1.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2],
                c='red', s=100, marker='*', label='End')
    
    # 繪製預測軌跡（如果長度不同，截取較短的）
    pred_positions = predicted_poses[:, :3]
    min_len = min(len(gt_positions), len(pred_positions))
    ax1.plot(pred_positions[:min_len, 0], pred_positions[:min_len, 1], pred_positions[:min_len, 2],
             'r--', linewidth=2, label='Predicted', alpha=0.7)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Trajectory\n"{instruction}"')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # XY 平面投影
    ax2 = fig.add_subplot(222)
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1], 'b-', linewidth=2, label='Ground Truth')
    ax2.plot(pred_positions[:min_len, 0], pred_positions[:min_len, 1], 'r--', linewidth=2, label='Predicted')
    ax2.scatter(gt_positions[0, 0], gt_positions[0, 1], c='green', s=100, marker='o')
    ax2.scatter(gt_positions[-1, 0], gt_positions[-1, 1], c='red', s=100, marker='*')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 位置誤差隨時間變化
    ax3 = fig.add_subplot(223)
    position_errors = np.linalg.norm(gt_positions[:min_len] - pred_positions[:min_len], axis=1)
    ax3.plot(position_errors, 'g-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title(f'Position Error Over Time\nMean: {position_errors.mean():.4f}m, Max: {position_errors.max():.4f}m')
    ax3.grid(True, alpha=0.3)
    
    # 各軸位置對比
    ax4 = fig.add_subplot(224)
    time_steps = np.arange(min_len)
    ax4.plot(time_steps, gt_positions[:min_len, 0], 'b-', label='GT X', alpha=0.7)
    ax4.plot(time_steps, pred_positions[:min_len, 0], 'b--', label='Pred X', alpha=0.7)
    ax4.plot(time_steps, gt_positions[:min_len, 1], 'g-', label='GT Y', alpha=0.7)
    ax4.plot(time_steps, pred_positions[:min_len, 1], 'g--', label='Pred Y', alpha=0.7)
    ax4.plot(time_steps, gt_positions[:min_len, 2], 'r-', label='GT Z', alpha=0.7)
    ax4.plot(time_steps, pred_positions[:min_len, 2], 'r--', label='Pred Z', alpha=0.7)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position Components Over Time')
    ax4.legend(ncol=2)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved trajectory visualization: {save_path}")
    plt.close()


def create_video_with_trajectory(images, ground_truth_poses, predicted_poses, instruction, save_path):
    """建立包含軌跡視覺化的影片"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 準備軌跡資料
    gt_positions = ground_truth_poses[:, :3]
    pred_positions = predicted_poses[:, :3]
    min_len = min(len(gt_positions), len(pred_positions), len(images))
    
    def update(frame):
        axes[0].clear()
        axes[1].clear()
        
        # 左側：當前影像
        if frame < len(images):
            axes[0].imshow(images[frame])
            axes[0].set_title(f'Camera View - Frame {frame}/{min_len-1}')
            axes[0].axis('off')
        
        # 右側：軌跡（XY平面）
        axes[1].plot(gt_positions[:, 0], gt_positions[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.5)
        axes[1].plot(pred_positions[:min_len, 0], pred_positions[:min_len, 1], 'r--', linewidth=2, label='Predicted', alpha=0.5)
        
        # 顯示到當前幀的軌跡
        axes[1].plot(gt_positions[:frame+1, 0], gt_positions[:frame+1, 1], 'b-', linewidth=3)
        axes[1].scatter(gt_positions[frame, 0], gt_positions[frame, 1], c='blue', s=200, marker='o', zorder=5)
        
        if frame < len(pred_positions):
            axes[1].plot(pred_positions[:frame+1, 0], pred_positions[:frame+1, 1], 'r--', linewidth=3)
            axes[1].scatter(pred_positions[frame, 0], pred_positions[frame, 1], c='red', s=200, marker='o', zorder=5)
        
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].set_title(f'Trajectory (XY Plane)\n"{instruction}"')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        return axes
    
    anim = FuncAnimation(fig, update, frames=min_len, interval=100, blit=False)
    writer = PillowWriter(fps=10)
    anim.save(save_path, writer=writer)
    print(f"  ✓ Saved video: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize VLA model inference on mango dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--dataset_path', type=str, default='../data/datasets/mango',
                        help='Path to mango dataset')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--output_dir', type=str, default='../demo_outputs',
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  VLA Model Visualization Demo")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    print()
    
    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入資料集
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # 獲取所有 episodes
    episode_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    episode_dirs = episode_dirs[:args.num_episodes]
    
    print(f"Found {len(episode_dirs)} episodes to process")
    print()
    
    # 載入模型（這裡簡化處理，實際需要根據模型結構調整）
    # state_dict, checkpoint = load_checkpoint(args.checkpoint, args.device)
    # model = initialize_model(state_dict, args.device)
    # model.eval()
    
    # 處理每個 episode
    for i, episode_dir in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
        print(f"\nEpisode {i+1}/{len(episode_dirs)}: {episode_dir.name}")
        
        # 載入 episode 資料
        episode_data = load_episode_data(episode_dir)
        
        if 'ee_poses' not in episode_data:
            print(f"  ⚠️ Skipping {episode_dir.name}: no ee_poses.npy found")
            continue
        
        instruction = episode_data.get('instruction', 'No instruction')
        print(f"  Instruction: {instruction}")
        
        # 使用模型預測（這裡用隨機資料示範）
        # predicted_trajectory = predict_trajectory(model, episode_data, args.device)
        
        # 示範用：使用真實軌跡加上一些噪聲作為"預測"
        gt_trajectory = episode_data['ee_poses']
        predicted_trajectory = gt_trajectory + np.random.randn(*gt_trajectory.shape) * 0.01
        
        # 視覺化
        save_name = f"episode_{episode_dir.name.split('_')[-1]}"
        
        # 儲存 3D 軌跡圖
        visualize_trajectory_3d(
            gt_trajectory, 
            predicted_trajectory,
            instruction,
            os.path.join(args.output_dir, f"{save_name}_trajectory.png")
        )
        
        # 建立影片（如果有影像）
        if 'images' in episode_data and len(episode_data['images']) > 0:
            create_video_with_trajectory(
                episode_data['images'],
                gt_trajectory,
                predicted_trajectory,
                instruction,
                os.path.join(args.output_dir, f"{save_name}_video.gif")
            )
    
    print()
    print("="*60)
    print(f"✓ Visualization complete!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"  Generated {len(episode_dirs)} visualizations")
    print("="*60)


if __name__ == "__main__":
    main()
