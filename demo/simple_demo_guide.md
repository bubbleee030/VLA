# VLA 機器人控制 Demo 指南

## 專案概述
這個專案讓你可以：
1. 訓練一個視覺-語言-動作(VLA)模型
2. 給予文字指令
3. 讓機器手臂移動到目標位置（先在視覺化環境測試，不需實體機器人）

## 現有資料集
你已經有 **mango 資料集**，包含約 181 個 episodes，每個 episode 包含：
- `camera1/`, `camera2/`: 相機影像
- `gelsight/`: 觸覺感測影像
- `ee_poses.npy`: 末端執行器位置（位置+旋轉）
- `gripper_pos.npy`: 夾爪開合狀態
- `instruction.txt`: 文字指令
- `instruction_embedding.pt`: 指令的嵌入向量

## 快速開始步驟

### 步驟 1: 準備環境
```bash
cd /home/cmwang16/VLA
# 確保已安裝所有依賴
```

### 步驟 2: 小規模訓練測試 (推薦先做這個)
使用小量資料快速訓練，確認流程可行：

```bash
bash quick_train_demo.sh
```

這會訓練 1000 steps，大約 10-30 分鐘完成。

### 步驟 3: 視覺化推論
訓練完成後，使用模型進行推論並視覺化：

```bash
python visualize_inference.py \
    --checkpoint ./outputs/demo_quick/last.ckpt \
    --dataset_path ./data/datasets/mango \
    --num_episodes 5 \
    --output_dir ./demo_outputs
```

這會：
- 載入訓練好的模型
- 從 mango 資料集選擇 5 個 episodes
- 顯示模型預測的軌跡 vs 真實軌跡
- 儲存視覺化影片到 `./demo_outputs`

### 步驟 4: 互動式指令測試
```bash
python interactive_demo.py \
    --checkpoint ./outputs/demo_quick/last.ckpt
```

你可以輸入自己的指令，看模型如何規劃動作。

## 訓練資料說明

### 資料格式
每個 episode 是一次完整的操作示範：
- **輸入**: 相機影像 + 文字指令 + 當前機器人狀態
- **輸出**: 未來的動作序列（位置、旋轉、夾爪）

### 訓練目標
模型學習：
```
給定: "把芒果移到左邊" (文字) + 當前影像 + 當前位置
預測: 接下來 64 步的動作 (action_chunk_size=64)
```

### 動作空間 (10維)
```python
action = [
    ee_pos_x,      # 0-2: 末端執行器 XYZ 位置
    ee_pos_y, 
    ee_pos_z,
    ee_ori_1,      # 3-8: 6D 旋轉表示
    ee_ori_2,
    ee_ori_3,
    ee_ori_4,
    ee_ori_5,
    ee_ori_6,
    gripper_open   # 9: 夾爪開合 (0=閉合, 1=張開)
]
```

## 完整訓練 (可選)
如果快速測試成功，可以進行完整訓練：

```bash
bash finetune.sh
```

這需要較長時間（數小時到數天），但會得到更好的效果。

## 檔案說明
- `configs/base.yaml`: 基礎配置
- `configs/finetune_datasets.json`: 訓練資料集列表
- `main.py`: 訓練主程式
- `models/rdt_runner.py`: RDT 模型實作

## 常見問題

### Q: 我可以用自己的指令嗎？
A: 可以！訓練後，模型可以理解類似的指令。你也可以：
1. 修改 mango 資料集中的 `instruction.txt`
2. 收集新的示範資料

### Q: 如何收集新資料？
A: 你需要：
1. 記錄機器人操作過程
2. 儲存影像、機器人狀態、動作
3. 轉換成相同的資料格式

### Q: 不訓練可以直接用嗎？
A: 可以！你可以：
1. 載入現有 checkpoint: `./checkpoints/mango/` 或 `./outputs/`
2. 直接運行視覺化腳本

## 下一步
1. ✅ 運行快速訓練測試
2. ✅ 視覺化結果
3. ✅ 測試不同指令
4. 🔄 收集更多資料（可選）
5. 🔄 部署到實體機器人（之後）
