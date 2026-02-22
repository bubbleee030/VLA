#!/usr/bin/env python3
"""
簡單資料集視覺化工具 - 不需要訓練模型
直接視覺化 mango 資料集中的機器人軌跡和影像

Usage:
    python simple_visualize_data.py
    python simple_visualize_data.py --num_episodes 10
"""

import argparse
import os
from pathlib import Path
import numpy as np
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


def load_episode(episode_path):
    """載入單個 episode 的所有資料"""
    episode_path = Path(episode_path)
    data = {}
    
    # 載入末端執行器位姿
    ee_path = episode_path / "ee_poses.npy"
    if ee_path.exists():
        data['ee_poses'] = np.load(ee_path)
    
    # 載入夾爪狀態
    gripper_path = episode_path / "gripper_pos.npy"
    if gripper_path.exists():
        data['gripper_pos'] = np.load(gripper_path)
    
    # 載入指令
    instruction_path = episode_path / "instruction.txt"
    if instruction_path.exists():
        with open(instruction_path, 'r', encoding='utf-8') as f:
            data['instruction'] = f.read().strip()
    else:
        data['instruction'] = "未知指令"
    
    # 載入影像（camera1）
    camera1_path = episode_path / "camera1"
    if camera1_path.exists():
        # 支援 .png 和 .jpg 格式
        image_files = sorted(list(camera1_path.glob("*.png")) + list(camera1_path.glob("*.jpg")))
        if image_files:
            images = []
            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
            if images:
                data['images'] = np.array(images)
    
    # 載入觸覺影像（gelsight）
    gelsight_path = episode_path / "gelsight"
    if gelsight_path.exists():
        # 支援 .png 和 .jpg 格式
        image_files = sorted(list(gelsight_path.glob("*.png")) + list(gelsight_path.glob("*.jpg")))
        if image_files:
            tactile_images = []
            for img_file in image_files[:10]:  # 只載入前10張
                img = cv2.imread(str(img_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tactile_images.append(img)
            if tactile_images:
                data['tactile_images'] = np.array(tactile_images)
    
    return data


def plot_trajectory_overview(data, save_path):
    """繪製軌跡總覽圖"""
    fig = plt.figure(figsize=(18, 10))
    
    instruction = data.get('instruction', '未知指令')
    ee_poses = data['ee_poses']
    positions = ee_poses[:, :3]  # XYZ 位置
    gripper = data.get('gripper_pos', np.zeros(len(ee_poses)))
    
    # 1. 3D 軌跡
    ax1 = fig.add_subplot(231, projection='3d')
    
    # 顏色映射（時間進展）
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
    for i in range(len(positions) - 1):
        ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                color=colors[i], linewidth=2)
    
    ax1.scatter(*positions[0], c='green', s=200, marker='o', label='起點', 
                edgecolors='black', linewidths=2)
    ax1.scatter(*positions[-1], c='red', s=200, marker='*', label='終點', 
                edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax1.set_title('3D 軌跡', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. XY 平面
    ax2 = fig.add_subplot(232)
    for i in range(len(positions) - 1):
        ax2.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=colors[i], linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=150, marker='o', 
                edgecolors='black', linewidths=2)
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=150, marker='*', 
                edgecolors='black', linewidths=2)
    
    # 添加方向箭頭
    step = max(len(positions) // 10, 1)
    for i in range(0, len(positions) - 1, step):
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]
        if abs(dx) > 0.001 or abs(dy) > 0.001:
            ax2.arrow(positions[i, 0], positions[i, 1], dx, dy,
                     head_width=0.015, head_length=0.015, fc=colors[i], ec=colors[i], alpha=0.5)
    
    ax2.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax2.set_title('XY 平面投影', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. XZ 平面
    ax3 = fig.add_subplot(233)
    for i in range(len(positions) - 1):
        ax3.plot(positions[i:i+2, 0], positions[i:i+2, 2], color=colors[i], linewidth=2)
    ax3.scatter(positions[0, 0], positions[0, 2], c='green', s=150, marker='o',
                edgecolors='black', linewidths=2)
    ax3.scatter(positions[-1, 0], positions[-1, 2], c='red', s=150, marker='*',
                edgecolors='black', linewidths=2)
    ax3.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Z (m)', fontsize=11, fontweight='bold')
    ax3.set_title('XZ 平面投影', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 位置隨時間變化
    ax4 = fig.add_subplot(234)
    time_steps = np.arange(len(positions))
    ax4.plot(time_steps, positions[:, 0], 'r-', label='X', linewidth=2)
    ax4.plot(time_steps, positions[:, 1], 'g-', label='Y', linewidth=2)
    ax4.plot(time_steps, positions[:, 2], 'b-', label='Z', linewidth=2)
    ax4.set_xlabel('時間步', fontsize=11, fontweight='bold')
    ax4.set_ylabel('位置 (m)', fontsize=11, fontweight='bold')
    ax4.set_title('位置隨時間變化', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 速度
    ax5 = fig.add_subplot(235)
    velocities = np.diff(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    ax5.plot(time_steps[:-1], speeds, 'purple', linewidth=2)
    ax5.set_xlabel('時間步', fontsize=11, fontweight='bold')
    ax5.set_ylabel('速度 (m/step)', fontsize=11, fontweight='bold')
    ax5.set_title(f'移動速度\n平均: {speeds.mean():.4f} m/step', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 夾爪狀態
    ax6 = fig.add_subplot(236)
    ax6.plot(time_steps, gripper, 'orange', linewidth=2)
    ax6.fill_between(time_steps, 0, gripper, alpha=0.3, color='orange')
    ax6.set_xlabel('時間步', fontsize=11, fontweight='bold')
    ax6.set_ylabel('夾爪開合度', fontsize=11, fontweight='bold')
    ax6.set_title('夾爪狀態 (0=閉合, 1=張開)', fontsize=12, fontweight='bold')
    ax6.set_ylim([-0.1, 1.1])
    ax6.grid(True, alpha=0.3)
    
    # 總標題
    distance = np.sum(speeds)
    fig.suptitle(f'Episode 軌跡分析\n指令: "{instruction}"\n'
                 f'時長: {len(positions)} 步 | 總距離: {distance:.3f} m | '
                 f'起點: ({positions[0,0]:.2f}, {positions[0,1]:.2f}, {positions[0,2]:.2f}) → '
                 f'終點: ({positions[-1,0]:.2f}, {positions[-1,1]:.2f}, {positions[-1,2]:.2f})',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_video(data, save_path):
    """建立動畫影片"""
    if 'images' not in data or len(data['images']) == 0:
        print("  ⚠️ 沒有影像資料，跳過影片生成")
        return
    
    images = data['images']
    positions = data['ee_poses'][:, :3]
    gripper = data.get('gripper_pos', np.zeros(len(positions)))
    instruction = data.get('instruction', '未知指令')
    
    # 確保長度一致
    min_len = min(len(images), len(positions))
    images = images[:min_len]
    positions = positions[:min_len]
    gripper = gripper[:min_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    def update(frame):
        axes[0].clear()
        axes[1].clear()
        
        # 左側：影像
        axes[0].imshow(images[frame])
        axes[0].set_title(f'相機視角 - 第 {frame}/{min_len-1} 幀\n夾爪狀態: {gripper[frame]:.2f}',
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 右側：軌跡
        colors = plt.cm.viridis(np.linspace(0, 1, min_len))
        
        # 完整軌跡（淡色）
        axes[1].plot(positions[:, 0], positions[:, 1], 'gray', linewidth=1, alpha=0.3)
        
        # 已完成軌跡（亮色）
        for i in range(frame):
            axes[1].plot(positions[i:i+2, 0], positions[i:i+2, 1], 
                        color=colors[i], linewidth=3)
        
        # 當前位置
        axes[1].scatter(positions[frame, 0], positions[frame, 1], 
                       c='red', s=300, marker='o', edgecolors='black', linewidths=3, zorder=10)
        
        # 起點和終點標記
        axes[1].scatter(positions[0, 0], positions[0, 1], 
                       c='green', s=200, marker='o', edgecolors='black', linewidths=2, alpha=0.7)
        axes[1].scatter(positions[-1, 0], positions[-1, 1], 
                       c='blue', s=200, marker='*', edgecolors='black', linewidths=2, alpha=0.7)
        
        axes[1].set_xlabel('X (m)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        axes[1].set_title(f'XZ 平面軌跡\n"{instruction}"', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        # 添加位置文字
        axes[1].text(0.02, 0.98, f'位置: ({positions[frame,0]:.3f}, {positions[frame,1]:.3f}, {positions[frame,2]:.3f})',
                    transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    anim = FuncAnimation(fig, update, frames=min_len, interval=100, blit=False)
    writer = PillowWriter(fps=10)
    anim.save(save_path, writer=writer)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='簡單視覺化 mango 資料集')
    parser.add_argument('--dataset_path', type=str, default='../data/datasets/mango',
                       help='資料集路徑')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='要視覺化的 episode 數量')
    parser.add_argument('--output_dir', type=str, default='../data_visualization',
                       help='輸出目錄')
    parser.add_argument('--skip_videos', action='store_true',
                       help='跳過影片生成（節省時間）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  Mango 資料集視覺化工具")
    print("="*70)
    print(f"資料集路徑: {args.dataset_path}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"Episode 數量: {args.num_episodes}")
    print("="*70)
    print()
    
    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 獲取資料集路徑
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ 錯誤：找不到資料集 {dataset_path}")
        print(f"   請確認路徑是否正確")
        return
    
    # 獲取所有 episodes
    episode_dirs = sorted([d for d in dataset_path.iterdir() 
                          if d.is_dir() and d.name.startswith('episode_')])
    
    if not episode_dirs:
        print(f"❌ 錯誤：在 {dataset_path} 中找不到 episode 資料夾")
        return
    
    print(f"✓ 找到 {len(episode_dirs)} 個 episodes")
    episode_dirs = episode_dirs[:args.num_episodes]
    print(f"  將處理前 {len(episode_dirs)} 個 episodes")
    print()
    
    # 處理每個 episode
    for i, episode_dir in enumerate(tqdm(episode_dirs, desc="處理 episodes")):
        episode_num = episode_dir.name.split('_')[-1]
        print(f"\n處理 Episode {episode_num}:")
        
        # 載入資料
        try:
            data = load_episode(episode_dir)
        except Exception as e:
            print(f"  ❌ 載入失敗: {e}")
            continue
        
        if 'ee_poses' not in data:
            print(f"  ⚠️ 跳過：找不到 ee_poses.npy")
            continue
        
        instruction = data.get('instruction', '未知指令')
        num_steps = len(data['ee_poses'])
        has_images = 'images' in data and len(data['images']) > 0
        
        print(f"  指令: {instruction}")
        print(f"  步數: {num_steps}")
        print(f"  影像: {'✓' if has_images else '✗'}")
        
        # 生成軌跡圖
        trajectory_path = os.path.join(args.output_dir, f"episode_{episode_num}_trajectory.png")
        try:
            plot_trajectory_overview(data, trajectory_path)
            print(f"  ✓ 軌跡圖已儲存: {trajectory_path}")
        except Exception as e:
            print(f"  ❌ 軌跡圖生成失敗: {e}")
        
        # 生成影片
        if not args.skip_videos and has_images:
            video_path = os.path.join(args.output_dir, f"episode_{episode_num}_video.gif")
            try:
                print(f"  生成影片中...", end='', flush=True)
                create_video(data, video_path)
                print(f" ✓ 已儲存: {video_path}")
            except Exception as e:
                print(f" ❌ 失敗: {e}")
    
    print()
    print("="*70)
    print(f"✓ 視覺化完成！")
    print(f"  結果儲存在: {args.output_dir}")
    print(f"  共處理 {len(episode_dirs)} 個 episodes")
    print()
    print("💡 提示：")
    print("  - 軌跡圖顯示機器人的移動路徑")
    print("  - 影片結合了相機視角和軌跡動畫")
    print("  - 使用 --skip_videos 可以跳過影片生成以節省時間")
    print("="*70)


if __name__ == "__main__":
    main()
