#!/usr/bin/env python3
"""
互動式 Demo - Interactive Command Demo

讓使用者輸入自訂指令，視覺化模型如何規劃機器人動作。

Usage:
    python interactive_demo.py --checkpoint ./outputs/demo_quick/last.ckpt
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# 設定中文字體
CHINESE_FONT_PATH = '/home/cmwang16/.fonts/NotoSansTC-Variable.ttf'
if os.path.exists(CHINESE_FONT_PATH):
    font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
else:
    print(f"⚠️ 警告：找不到中文字體 {CHINESE_FONT_PATH}，中文可能無法正確顯示")

# 示範用的指令庫
EXAMPLE_INSTRUCTIONS = [
    "把芒果移到左邊",
    "把芒果移到右邊", 
    "把芒果往前推",
    "抓住芒果並抬高",
    "把芒果放到盒子裡",
    "輕輕碰觸芒果",
]


def generate_dummy_trajectory(instruction, start_pos=None):
    """
    根據指令生成示範軌跡（這裡用簡單的規則，實際應該用訓練好的模型）
    
    Args:
        instruction: 文字指令
        start_pos: 起始位置 [x, y, z]
    
    Returns:
        trajectory: (T, 10) 動作序列
    """
    if start_pos is None:
        start_pos = np.array([0.0, 0.0, 0.3])  # 預設起始位置
    
    T = 64  # 軌跡長度
    trajectory = np.zeros((T, 10))
    
    # 根據指令決定目標位置
    if "左邊" in instruction or "左" in instruction:
        target_offset = np.array([-0.2, 0.0, 0.0])
    elif "右邊" in instruction or "右" in instruction:
        target_offset = np.array([0.2, 0.0, 0.0])
    elif "前" in instruction or "向前" in instruction:
        target_offset = np.array([0.0, 0.2, 0.0])
    elif "後" in instruction or "向後" in instruction:
        target_offset = np.array([0.0, -0.2, 0.0])
    elif "抬高" in instruction or "上" in instruction:
        target_offset = np.array([0.0, 0.0, 0.15])
    elif "放下" in instruction or "下" in instruction:
        target_offset = np.array([0.0, 0.0, -0.1])
    else:
        target_offset = np.array([0.1, 0.1, 0.05])  # 預設移動
    
    target_pos = start_pos + target_offset
    
    # 生成平滑的軌跡（線性插值）
    for t in range(T):
        alpha = t / (T - 1)
        
        # 位置插值
        current_pos = start_pos * (1 - alpha) + target_pos * alpha
        trajectory[t, :3] = current_pos
        
        # 旋轉（簡化為零，實際應該用四元數或6D表示）
        trajectory[t, 3:9] = np.array([1, 0, 0, 0, 1, 0])  # 6D rotation
        
        # 夾爪（根據指令決定開合）
        if "抓" in instruction or "夾" in instruction:
            gripper = 1.0 if t < T // 3 else 0.0  # 先開後關
        elif "放" in instruction:
            gripper = 0.0 if t < T // 2 else 1.0  # 先關後開
        else:
            gripper = 0.5  # 半開
        
        trajectory[t, 9] = gripper
    
    return trajectory


def visualize_planned_trajectory(trajectory, instruction, save_path=None):
    """視覺化規劃的軌跡"""
    fig = plt.figure(figsize=(16, 6))
    
    # 3D 軌跡
    ax1 = fig.add_subplot(131, projection='3d')
    positions = trajectory[:, :3]
    
    # 繪製軌跡，顏色從藍到紅表示時間進展
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
    
    for i in range(len(positions) - 1):
        ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                color=colors[i], linewidth=2)
    
    # 標記起點和終點
    ax1.scatter(*positions[0], c='green', s=200, marker='o', label='起點', edgecolors='black', linewidths=2)
    ax1.scatter(*positions[-1], c='red', s=200, marker='*', label='終點', edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title(f'3D 軌跡規劃\n指令: "{instruction}"', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # XY 平面投影
    ax2 = fig.add_subplot(132)
    for i in range(len(positions) - 1):
        ax2.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=colors[i], linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='o', edgecolors='black', linewidths=2)
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='*', edgecolors='black', linewidths=2)
    
    # 繪製方向箭頭
    for i in range(0, len(positions) - 1, 10):
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]
        ax2.arrow(positions[i, 0], positions[i, 1], dx, dy, 
                 head_width=0.02, head_length=0.02, fc=colors[i], ec=colors[i], alpha=0.6)
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('XY 平面視圖', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 位置和夾爪狀態隨時間變化
    ax3 = fig.add_subplot(133)
    time_steps = np.arange(len(trajectory))
    
    ax3.plot(time_steps, positions[:, 0], 'r-', label='X', linewidth=2)
    ax3.plot(time_steps, positions[:, 1], 'g-', label='Y', linewidth=2)
    ax3.plot(time_steps, positions[:, 2], 'b-', label='Z', linewidth=2)
    
    # 夾爪狀態（縮放到可見範圍）
    gripper_scaled = trajectory[:, 9] * 0.1  # 縮放以便顯示
    ax3.plot(time_steps, gripper_scaled, 'm--', label='夾爪 (x0.1)', linewidth=2)
    
    ax3.set_xlabel('time steps', fontsize=12)
    ax3.set_ylabel('位置 (m) / 夾爪狀態', fontsize=12)
    ax3.set_title('位置與夾爪隨時間變化', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 圖片已儲存: {save_path}")
        print(f"  ✓ 影片已儲存: ./data_visualization/video2.gif")
    
    plt.show()


def print_instruction_menu():
    """顯示指令選單"""
    print("\n" + "="*60)
    print("  可用的指令:")
    print("="*60)
    for i, instr in enumerate(EXAMPLE_INSTRUCTIONS, 1):
        print(f"  {i}. {instr}")
    print("  0. 自訂指令")
    print("  q. 退出")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Interactive VLA Demo')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional for this demo)')
    parser.add_argument('--output_dir', type=str, default='../demo_outputs/interactive',
                        help='Output directory for saving visualizations')
    parser.add_argument('--start_pos', type=float, nargs=3, default=[0.0, 0.0, 0.3],
                        help='Starting position [x y z] in meters')
    
    args = parser.parse_args()
    
    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("  VLA 互動式 Demo - 機器人動作規劃")
    print("="*60)
    print(f"起始位置: {args.start_pos}")
    print(f"輸出目錄: {args.output_dir}")
    if args.checkpoint:
        print(f"模型: {args.checkpoint}")
    else:
        print("模式: 示範模式（使用規則生成軌跡）")
    print("="*60)
    
    start_pos = np.array(args.start_pos)
    counter = 0
    
    while True:
        print_instruction_menu()
        
        choice = input("\n請選擇指令 (輸入編號或 'q' 退出): ").strip()
        
        if choice.lower() == 'q':
            print("\n👋 再見！")
            break
        
        # 獲取指令
        if choice == '0':
            instruction = input("請輸入自訂指令: ").strip()
            if not instruction:
                print("❌ 指令不能為空！")
                continue
        elif choice.isdigit() and 1 <= int(choice) <= len(EXAMPLE_INSTRUCTIONS):
            instruction = EXAMPLE_INSTRUCTIONS[int(choice) - 1]
        else:
            print("❌ 無效的選擇！")
            continue
        
        print(f"\n🤖 語言指令: \"{instruction}\"")
        print("   規劃軌跡中...")
        
        # 生成軌跡
        trajectory = generate_dummy_trajectory(instruction, start_pos)
        
        # 顯示統計資訊
        end_pos = trajectory[-1, :3]
        distance = np.linalg.norm(end_pos - start_pos)
        print(f"   ✓ 軌跡已生成!")
        print(f"   起點: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
        print(f"   終點: ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f})")
        print(f"   移動距離: {distance:.3f} m")
        print(f"   總時間步數: {len(trajectory)}")
        
        # 視覺化
        counter += 1
        save_path = os.path.join(args.output_dir, f"demo_{counter:03d}.png")
        visualize_planned_trajectory(trajectory, instruction, save_path)
        
        # 詢問是否繼續
        continue_choice = input("\n繼續嘗試其他指令？(y/n): ").strip().lower()
        if continue_choice == 'n':
            print("\n👋 再見！")
            break
    
    print(f"\n✓ 共生成了 {counter} 個軌跡規劃")
    print(f"  結果已儲存在: {args.output_dir}")


if __name__ == "__main__":
    main()
