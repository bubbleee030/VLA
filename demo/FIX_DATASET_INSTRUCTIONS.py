#!/usr/bin/env python3
"""
修正数据集：为不同的视频分配不同的指令
当前问题：所有4个episodes用同一个指令，导致任务混淆
解决方案：分别为bigboard和toolbox任务分配不同指令
"""

import json
from pathlib import Path

# 修正元数据
own_processed = Path("../data/datasets/own_processed")

tasks = {
    "episode_0": {"video": "bigboard.mp4", "instruction": "移动到电路板区域"},
    "episode_1": {"video": "bigboard_near.mp4", "instruction": "靠近电路板进行操作"},
    "episode_2": {"video": "toolbox.mp4", "instruction": "移动到工具箱区域"},
    "episode_3": {"video": "toolbox_near.mp4", "instruction": "靠近工具箱进行操作"},
}

print("=" * 70)
print("  修正数据集指令")
print("=" * 70)

for ep_name, task_info in tasks.items():
    metadata_file = own_processed / ep_name / "metadata.json"
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    old_instruction = metadata['instruction']
    metadata['instruction'] = task_info['instruction']
    metadata['video_source'] = task_info['video']
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ {ep_name}")
    print(f"  视频：{task_info['video']}")
    print(f"  旧指令：{old_instruction[:40]}...")
    print(f"  新指令：{task_info['instruction']}")

print("\n" + "=" * 70)
print("  修正完成！")
print("=" * 70)
print("\n现在可以安全地开始训练了：")
print("  bash quick_train_demo.sh")
print("  或")
print("  bash train_with_own_data.sh")
