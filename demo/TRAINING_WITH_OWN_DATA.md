# 使用自己的資料進行訓練

## 🎯 快速開始

### 步驟 1：準備資料

**1.1 轉換手臂位置資料**

將 `arm_position.docx` 轉換為 CSV 格式：

```csv
time,x,y,z,qx,qy,qz,qw,gripper
0.0,0.5,0.2,0.3,0,0,0,1,0.0
0.1,0.51,0.2,0.3,0,0,0,1,0.1
0.2,0.52,0.21,0.31,0,0,0,1,0.2
```

**欄位說明：**
- `time`: 時間戳（秒）
- `x, y, z`: 手臂末端位置（公尺）
- `qx, qy, qz, qw`: 旋轉四元數
- `gripper`: 夾爪開合度（0=關閉，1=打開）

儲存為：`/home/cmwang16/VLA/data/datasets/own/arm_position.csv`

**1.2 運行資料轉換腳本**

```bash
cd /home/cmwang16/VLA/demo
python3 prepare_own_dataset.py \
    --input_dir ../data/datasets/own \
    --output_dir ../data/datasets/own_processed
```

這會：
- 從影片中提取影格
- 對齊影格和手臂資料
- 建立訓練所需的 episode 資料夾結構｀

**1.3 檢查生成的資料**

```bash
# 查看資料結構
ls -lh ../data/datasets/own_processed/

# 應該會看到：
# episode_0/
# episode_1/
# episode_2/
# episode_3/
# dataset_info.json
```

### 步驟 2：視覺化檢查

```bash
# 視覺化您的資料（確認資料品質）
python3 simple_visualize_data.py \
    --dataset_path ../data/datasets/own_processed \
    --num_episodes 4
```

檢查生成的圖片：
- 軌跡是否平滑？
- 影格和動作是否對齊？
- 指令是否正確？

### 步驟 3：更新訓練配置

**3.1 添加資料集到配置檔案**

編輯 `../configs/finetune_datasets.json`：

```json
{
  "own_dataset": {
    "dataset_path": "data/datasets/own_processed",
    "dataset_type": "directory",
    "num_episodes": 4
  },
  "mango": {
    "dataset_path": "data/datasets/mango",
    "dataset_type": "directory",
    "num_episodes": 181
  }
}
```

**3.2 設定採樣權重**

編輯 `../configs/finetune_sample_weights.json`：

```json
{
  "own_dataset": 0.5,
  "mango": 0.5
}
```

這表示訓練時會平均從兩個資料集採樣。

### 步驟 4：開始訓練

**選項 1：快速測試（推薦先用這個）**

```bash
bash quick_train_demo.sh
```

配置：
- 訓練步數：1000
- 約 20 分鐘
- 用於測試資料是否正確

**選項 2：完整訓練**

```bash
cd /home/cmwang16/VLA
bash finetune.sh
```

配置：
- 訓練步數：40000
- 約 10-15 小時
- 用於實際部署的模型

### 步驟 5：監控訓練

**查看日誌：**

```bash
# 即時查看
tail -f ../outputs/demo_quick/log.txt

# 查看訓練進度
grep "Loss" ../outputs/demo_quick/log.txt
```

**使用 TensorBoard：**

```bash
tensorboard --logdir ../outputs/demo_quick/
# 在瀏覽器開啟：http://localhost:6006
```

### 步驟 6：視覺化結果

```bash
# 找到最新的 checkpoint
CHECKPOINT=$(ls -t ../outputs/demo_quick/*.ckpt | head -1)

# 視覺化模型預測
python3 visualize_inference.py \
    --checkpoint $CHECKPOINT \
    --dataset_path ../data/datasets/own_processed \
    --num_episodes 4
```

查看結果：
- `../demo_outputs/episode_X_trajectory.png`
- `../demo_outputs/episode_X_video.gif`

### 步驟 7：互動式測試

```bash
python3 interactive_demo.py --checkpoint $CHECKPOINT
```

輸入測試指令：
- "移動到大板子"
- "靠近大板子"
- "移動到工具箱"
- "靠近工具箱"

---

## 📊 資料品質檢查清單

在訓練前，確認：

- [ ] CSV 格式正確（9 個欄位）
- [ ] 時間戳遞增且均勻
- [ ] 位置在合理範圍內（例如 0.2-0.8 公尺）
- [ ] 四元數已正規化（qx² + qy² + qz² + qw² = 1）
- [ ] 夾爪值在 0-1 之間
- [ ] 影片清晰且穩定
- [ ] 影片長度與資料點數量相符

---

## ⚙️ 訓練參數調整

如果訓練效果不好，可以調整：

### 學習率

```bash
# 在 quick_train_demo.sh 中修改
--learning_rate=1e-4  # 預設
--learning_rate=5e-5  # 更穩定但慢
--learning_rate=2e-4  # 更快但可能不穩定
```

### 批次大小

```bash
--train_batch_size=2  # 預設（RTX 5090）
--train_batch_size=4  # 如果 VRAM 足夠
--train_batch_size=1  # 如果 OOM
```

### 訓練步數

```bash
--max_train_steps=1000   # 快速測試
--max_train_steps=5000   # 中等訓練
--max_train_steps=40000  # 完整訓練
```

### 資料增強

```bash
--image_aug              # 啟用影像增強（預設）
--no_image_aug           # 禁用（如果資料少）
```

---

## 🔧 常見問題

### Q1: 資料點數量和影格數不一致怎麼辦？

A: `prepare_own_dataset.py` 會自動處理：
- 資料點多 → 重採樣
- 資料點少 → 插值

### Q2: 訓練時出現 OOM（記憶體不足）？

A: 降低批次大小：
```bash
--train_batch_size=1
--gradient_accumulation_steps=4  # 保持有效批次大小
```

### Q3: Loss 不下降？

可能原因：
1. 學習率太高或太低 → 調整 learning_rate
2. 資料品質問題 → 檢查視覺化結果
3. 資料太少 → 增加更多 episodes
4. 需要更長訓練 → 增加 max_train_steps

### Q4: 如何只用自己的資料訓練？

修改 `finetune_sample_weights.json`：
```json
{
  "own_dataset": 1.0,
  "mango": 0.0
}
```

### Q5: 如何加入更多影片？

1. 放置影片到 `data/datasets/own/`
2. 更新 `prepare_own_dataset.py` 中的 `video_configs`：
   ```python
   video_configs = [
       {"video": "bigboard.mp4", "instruction": "移動到大板子"},
       {"video": "bigboard_near.mp4", "instruction": "靠近大板子"},
       {"video": "toolbox.mp4", "instruction": "移動到工具箱"},
       {"video": "toolbox_near.mp4", "instruction": "靠近工具箱"},
       {"video": "new_task.mp4", "instruction": "新任務描述"},  # 新增
   ]
   ```
3. 重新運行 `prepare_own_dataset.py`

---

## 📈 預期效果

### 資料集大小 vs 效果

| Episodes | 訓練步數 | 預期效果 |
|----------|----------|----------|
| 4        | 1000     | 能複現訓練資料 |
| 4        | 5000     | 開始泛化 |
| 10+      | 10000    | 較好泛化能力 |
| 50+      | 40000    | 穩定部署 |

### 建議

- **先用少量資料快速迭代**：4 個 episodes × 1000 steps
- **驗證流程正確後收集更多資料**：10-20 個 episodes
- **完整訓練前再次視覺化檢查**：確保資料品質

---

## 🎯 進階：混合多個資料集

如果您有多種任務：

```json
// finetune_datasets.json
{
  "own_grasp": {"dataset_path": "data/datasets/own_grasp", "num_episodes": 10},
  "own_place": {"dataset_path": "data/datasets/own_place", "num_episodes": 8},
  "own_push": {"dataset_path": "data/datasets/own_push", "num_episodes": 5},
  "mango": {"dataset_path": "data/datasets/mango", "num_episodes": 181}
}

// finetune_sample_weights.json
{
  "own_grasp": 0.3,
  "own_place": 0.3,
  "own_push": 0.2,
  "mango": 0.2
}
```

這樣模型會學習多種技能！

---

## 📞 需要幫助？

如果遇到問題：

1. **檢查日誌**：`../outputs/demo_quick/log.txt`
2. **視覺化資料**：確認資料品質
3. **減少批次大小**：避免 OOM
4. **降低學習率**：如果 loss 震盪
5. **增加訓練步數**：如果還在下降

祝訓練順利！🚀
