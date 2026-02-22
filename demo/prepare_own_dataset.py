#!/usr/bin/env python3
"""
將您自己錄製的資料轉換成訓練格式

使用方法：
    python prepare_own_dataset.py --input_dir ../data/datasets/own --output_dir ../data/datasets/own_processed
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import shutil

def euler_to_quaternion(rx, ry, rz):
    """
    將歐拉角（度數）轉換為四元數
    
    Args:
        rx, ry, rz: 旋轉角度（度數）
    
    Returns:
        [qx, qy, qz, qw]
    """
    # 將度數轉換為弧度
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)
    
    # 計算半角
    cy = np.cos(ry_rad * 0.5)
    sy = np.sin(ry_rad * 0.5)
    cp = np.cos(rz_rad * 0.5)
    sp = np.sin(rz_rad * 0.5)
    cr = np.cos(rx_rad * 0.5)
    sr = np.sin(rx_rad * 0.5)
    
    # ZYX 欧拉角順序
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cy * sp - sr * cp * sy
    
    return np.array([qx, qy, qz, qw])


def parse_arm_position(csv_path):
    """
    解析機器手臂位置數據
    
    支援兩種格式：
    1. 時間序列格式：time,x,y,z,qx,qy,qz,qw,gripper
    2. 步驟序列格式：Phase,Step,X,Y,Z,RX,RY,RZ,Note
    
    Returns:
        dict: {'timestamps': [], 'ee_poses': [], 'gripper_pos': [], 'instructions': []}
    """
    if not csv_path.exists():
        print(f"✗ 找不到 CSV 檔案：{csv_path}")
        return None
    
    print(f"✓ 找到 CSV 檔案：{csv_path}")
    df = pd.read_csv(csv_path)
    
    # 檢測格式
    if 'Phase' in df.columns:
        # 格式 2：步驟序列格式（您的格式）
        print("✓ 檢測到步驟序列格式（Phase, Step, X, Y, Z, RX, RY, RZ）")
        return parse_step_sequence_format(df)
    elif 'time' in df.columns:
        # 格式 1：時間序列格式
        print("✓ 檢測到時間序列格式（time, x, y, z, qx, qy, qz, qw, gripper）")
        return parse_time_sequence_format(df)
    else:
        print("✗ 無法識別 CSV 格式")
        print(f"  已有欄位：{list(df.columns)}")
        return None


def parse_step_sequence_format(df):
    """
    解析步驟序列格式（Phase, Step, X, Y, Z, RX, RY, RZ）
    
    這種格式包含任務階段和步驟信息，需要轉換為時間序列
    """
    print("\n  解析步驟序列格式...")
    
    # 轉換單位：mm → m（除以 1000）
    print("  ⚠️  單位轉換：mm → m（除以 1000）")
    X = df['X'].values / 1000.0
    Y = df['Y'].values / 1000.0
    Z = df['Z'].values / 1000.0
    RX = df['RX'].values
    RY = df['RY'].values
    RZ = df['RZ'].values
    
    # 生成時間戳（假設每步 0.1 秒）
    timestamps = np.arange(len(df)) * 0.1
    
    # 轉換歐拉角為四元數
    ee_poses = []
    for i in range(len(df)):
        quat = euler_to_quaternion(RX[i], RY[i], RZ[i])
        pose = np.concatenate([[X[i], Y[i], Z[i]], quat])
        ee_poses.append(pose)
    ee_poses = np.array(ee_poses)
    
    # 夾爪狀態提取：從 Note 欄位或預設為 0
    gripper_pos = np.zeros(len(df))
    if 'Note' in df.columns:
        for i, note in enumerate(df['Note'].values):
            if isinstance(note, str):
                if '夾取' in note or '夾爪夾' in note:
                    gripper_pos[i] = 1.0  # 夾爪關閉
                elif '放下' in note or '夾爪放' in note:
                    gripper_pos[i] = 0.0  # 夾爪打開
                elif '攤平' in note:
                    gripper_pos[i] = 0.0
    
    # 智慧生成指令（從 Phase 欄位）
    unique_phases = df['Phase'].unique()
    phase_to_instruction = {
        'Grip_Toolbox': '夾取工具箱',
        'Lift_Toolbox': '舉起工具箱',
        'Return_Ready_Toolbox': '返回預備位置',
        'Pick_PCB': '夾取電路板',
        'Lift_PCB': '舉起電路板',
        'Return_Ready_PCB': '返回預備位置',
        'Ready_Global': '移動到預備位置',
        'Home': '回到初始位置'
    }
    
    instruction = ' → '.join([phase_to_instruction.get(p, p) for p in unique_phases if p != 'Home'])
    if not instruction:
        instruction = '機器手臂操作序列'
    
    print(f"  自動生成指令：{instruction}")
    print(f"  數據統計：")
    print(f"    - 步驟數：{len(df)}")
    print(f"    - 位置範圍 X：{X.min():.3f} ~ {X.max():.3f} m")
    print(f"    - 位置範圍 Y：{Y.min():.3f} ~ {Y.max():.3f} m")
    print(f"    - 位置範圍 Z：{Z.min():.3f} ~ {Z.max():.3f} m")
    print(f"    - 旋轉範圍 RX：{RX.min():.1f}° ~ {RX.max():.1f}°")
    print(f"    - 旋轉範圍 RY：{RY.min():.1f}° ~ {RY.max():.1f}°")
    print(f"    - 旋轉範圍 RZ：{RZ.min():.1f}° ~ {RZ.max():.1f}°")
    
    return {
        'timestamps': timestamps,
        'ee_poses': ee_poses,
        'gripper_pos': gripper_pos,
        'instruction': instruction
    }


def parse_time_sequence_format(df):
    """
    解析時間序列格式（time, x, y, z, qx, qy, qz, qw, gripper）
    """
    print("\n  解析時間序列格式...")
    
    timestamps = df['time'].values if 'time' in df.columns else np.arange(len(df)) * 0.1
    
    # ee_poses: [x, y, z, qx, qy, qz, qw]
    column_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    if all(col in df.columns for col in column_names):
        ee_poses = df[column_names].values
    else:
        print(f"✗ 缺少必要欄位。已有：{list(df.columns)}")
        return None
    
    # gripper_pos: [gripper]
    gripper_pos = df['gripper'].values if 'gripper' in df.columns else np.zeros(len(df))
    
    # 默認指令
    instruction = '機器手臂軌跡'
    
    return {
        'timestamps': timestamps,
        'ee_poses': ee_poses,
        'gripper_pos': gripper_pos,
        'instruction': instruction
    }


def extract_frames_from_video(video_path, output_dir, target_fps=10):
    """
    從影片中提取影格
    
    Args:
        video_path: 影片檔案路徑
        output_dir: 輸出目錄
        target_fps: 目標 FPS（預設 10）
    
    Returns:
        int: 提取的影格數量
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  影片 FPS: {original_fps:.2f}, 總影格數: {total_frames}")
    
    # 計算採樣間隔
    frame_interval = max(1, int(original_fps / target_fps))
    
    frame_idx = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc=f"  提取影格") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # 儲存影格
                output_path = output_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"  ✓ 提取了 {saved_count} 個影格（每秒 {target_fps} 個）")
    return saved_count


def align_data(arm_data, num_frames, instruction):
    """
    對齊手臂資料和影片影格數量
    """
    timestamps = arm_data['timestamps']
    ee_poses = arm_data['ee_poses']
    gripper_pos = arm_data['gripper_pos']
    
    # 如果資料點比影格多，進行重採樣
    if len(timestamps) > num_frames:
        indices = np.linspace(0, len(timestamps)-1, num_frames, dtype=int)
        ee_poses = ee_poses[indices]
        gripper_pos = gripper_pos[indices]
        timestamps = timestamps[indices]
    elif len(timestamps) < num_frames:
        # 如果資料點比影格少，進行插值
        print(f"  ⚠️  資料點 ({len(timestamps)}) 少於影格數 ({num_frames})，將進行插值")
        new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_frames)
        
        ee_poses_interp = np.zeros((num_frames, ee_poses.shape[1]))
        for i in range(ee_poses.shape[1]):
            ee_poses_interp[:, i] = np.interp(new_timestamps, timestamps, ee_poses[:, i])
        
        gripper_pos = np.interp(new_timestamps, timestamps, gripper_pos)
        ee_poses = ee_poses_interp
        timestamps = new_timestamps
    
    return {
        'timestamps': timestamps,
        'ee_poses': ee_poses,
        'gripper_pos': gripper_pos,
        'instruction': instruction
    }


def create_episode(episode_dir, video_path, arm_data):
    """
    建立一個 episode 資料夾
    
    Args:
        episode_dir: episode 資料夾路徑
        video_path: 影片檔案路徑（可選，None 表示無影片）
        arm_data: 手臂軌跡數據
    """
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n處理 Episode: {episode_dir.name}")
    print(f"  指令: {arm_data['instruction']}")
    
    # 1. 提取影格（如果有影片）
    if video_path and video_path.exists():
        print(f"  影片: {video_path.name}")
        camera1_dir = episode_dir / "camera1"
        num_frames = extract_frames_from_video(video_path, camera1_dir, target_fps=10)
    else:
        print(f"  ⚠️  無影片，只保存軌跡數據")
        num_frames = len(arm_data['timestamps'])
    
    # 2. 對齊資料
    aligned_data = align_data(arm_data, num_frames, arm_data['instruction'])
    
    # 3. 儲存軌跡數據
    np.save(episode_dir / "ee_poses.npy", aligned_data['ee_poses'])
    np.save(episode_dir / "gripper_pos.npy", aligned_data['gripper_pos'])
    
    # 4. 儲存 metadata
    duration = aligned_data['timestamps'][-1] - aligned_data['timestamps'][0] if len(aligned_data['timestamps']) > 1 else 0
    metadata = {
        'instruction': arm_data['instruction'],
        'num_frames': num_frames,
        'num_steps': len(arm_data.get('timestamps', [])),
        'duration_seconds': float(duration),
        'video_source': str(video_path.name) if video_path else 'No video',
        'created_date': pd.Timestamp.now().isoformat()
    }
    
    with open(episode_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Episode 建立完成")
    print(f"    - 影格數: {num_frames}")
    print(f"    - 時長: {metadata['duration_seconds']:.2f} 秒")
    print(f"    - ee_poses: {aligned_data['ee_poses'].shape}")
    print(f"    - gripper_pos: {aligned_data['gripper_pos'].shape}")


def main():
    parser = argparse.ArgumentParser(description="準備您的訓練資料集")
    parser.add_argument("--input_dir", default="../data/datasets/own", help="輸入資料夾")
    parser.add_argument("--output_dir", default="../data/datasets/own_processed", help="輸出資料夾")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("  準備訓練資料集")
    print("=" * 70)
    print(f"輸入資料夾: {input_dir}")
    print(f"輸出資料夾: {output_dir}")
    print()
    
    # 1. 解析手臂位置資料（支援 CSV 格式）
    csv_file = input_dir / "arm_position.csv"
    docx_file = input_dir / "arm_position.docx"
    
    if csv_file.exists():
        arm_data = parse_arm_position(csv_file)
    elif docx_file.exists():
        print(f"⚠️  找到 {docx_file.name}，但需要 CSV 格式")
        print("   請轉換為 CSV 並儲存為 arm_position.csv")
        return
    else:
        print(f"✗ 找不到手臂位置數據檔案")
        print(f"  期望位置：{csv_file}")
        print("\n  支援的格式：")
        print("    1. arm_position.csv (步驟序列格式)")
        print("       Phase,Step,X,Y,Z,RX,RY,RZ,Note")
        print("    2. arm_position.csv (時間序列格式)")
        print("       time,x,y,z,qx,qy,qz,qw,gripper")
        return
    
    if arm_data is None:
        return
    
    print(f"✓ 成功解析手臂位置數據")
    print()
    
    # 2. 處理所有影片和建立 episodes
    # 由於您的 CSV 包含完整的軌跡，所以不需要單個影片對應
    # 如果有影片，可以配對使用；否則只用軌跡數據
    
    video_files = sorted(list(input_dir.glob("*.mp4")))
    
    print("=" * 70)
    print("  建立 Episodes")
    print("=" * 70)
    print(f"找到 {len(video_files)} 個影片檔案")
    print()
    
    if len(video_files) == 0:
        print("⚠️  沒有找到影片檔案")
        print("  將只使用軌跡數據建立 episode")
        
        episode_dir = output_dir / "episode_0"
        create_episode(episode_dir, None, arm_data)
        episode_count = 1
    else:
        # 如果有影片，為每個影片建立一個 episode
        # 使用相同的軌跡數據（適合連續記錄的情況）
        episode_count = 0
        for video_path in video_files:
            episode_dir = output_dir / f"episode_{episode_count}"
            create_episode(episode_dir, video_path, arm_data)
            episode_count += 1
    
    # 3. 建立資料集摘要
    print("\n" + "=" * 70)
    print("  建立資料集摘要")
    print("=" * 70)
    
    dataset_info = {
        "name": "own_dataset",
        "num_episodes": episode_count,
        "instruction": arm_data['instruction'],
        "data_format": "Step sequence format (Phase, Step, X, Y, Z, RX, RY, RZ)",
        "unit_converted": "mm → m",
        "created_date": pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 資料集建立完成！")
    print(f"  - Episodes 數量: {episode_count}")
    print(f"  - 共同指令: {arm_data['instruction']}")
    print(f"  - 輸出目錄: {output_dir}")
    print()
    print("=" * 70)
    print("  下一步")
    print("=" * 70)
    print("1. 檢查生成的資料：")
    print(f"   ls -lh {output_dir}/episode_*/")
    print()
    print("2. 視覺化資料：")
    print(f"   cd {Path(args.output_dir).parent.parent}/demo")
    print(f"   python3 simple_visualize_data.py --dataset_path {output_dir}")
    print()
    print("3. 訓練模型：")
    print(f"   bash quick_train_demo.sh")
    print()


if __name__ == "__main__":
    main()
