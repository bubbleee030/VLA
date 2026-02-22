Ｖ# 🤖 VLA 機器人控制 Demo - 總結文檔

## ✅ 已完成的工作

我已經為你建立了完整的 demo 系統，讓你可以：
1. ✅ 視覺化現有的機器人示範資料
2. ✅ 輸入指令並看到規劃的軌跡
3. ✅ 訓練自己的模型（可選）
4. ✅ 不需要實體機器人就能測試

## 📁 新建立的檔案

### 1. 文檔類
- **`quick_start_zh.md`** - 中文快速開始指南（**推薦先看這個**）
- **`simple_demo_guide.md`** - 完整的英文文檔
- **`README_DEMO.md`** - 本檔案（總結）

### 2. 可執行腳本
- **`simple_visualize_data.py`** - 視覺化資料集（不需要模型）⭐ 推薦先用
- **`interactive_demo.py`** - 互動式指令測試
- **`visualize_inference.py`** - 視覺化模型推論結果
- **`quick_train_demo.sh`** - 快速訓練腳本

## 🚀 立即開始（三選一）

### 選項 1: 最簡單 - 視覺化現有資料 ⭐ 推薦

```bash
cd /home/cmwang16/VLA

# 視覺化 5 個 episodes（不需要訓練）
python simple_visualize_data.py --num_episodes 5

# 結果會儲存在 ./data_visualization/
```

**你會看到：**
- 📊 3D 軌跡圖
- 📈 位置和速度分析
- 🎬 動畫影片（可選）

**時間：** 約 1-2 分鐘

---

### 選項 2: 互動式 - 輸入指令看規劃

```bash
cd /home/cmwang16/VLA

# 啟動互動式 demo
python interactive_demo.py
```

**操作：**
1. 選擇範例指令或輸入自己的指令
2. 系統會顯示 3D 軌跡規劃
3. 可以測試多個不同的指令

**範例指令：**
- "把芒果移到左邊"
- "抓住芒果並抬高"
- "把芒果往前推"

**時間：** 每個指令約 5 秒

---

### 選項 3: 訓練模型（需要較長時間）

```bash
cd /home/cmwang16/VLA

# 快速訓練（1000 steps）
bash quick_train_demo.sh

# 訓練完成後，視覺化結果
python visualize_inference.py \
    --checkpoint ./outputs/demo_quick/last.ckpt \
    --num_episodes 5
```

**時間：** 訓練約 10-30 分鐘（取決於 GPU）

---

## 📊 你的資料集

你有 **mango 資料集**，包含 **181 個示範 episodes**：

```
data/datasets/mango/
├── episode_0/
│   ├── camera1/          # 相機影像
│   ├── camera2/          # 另一個角度
│   ├── gelsight/         # 觸覺感測
│   ├── ee_poses.npy      # 末端執行器位置（XYZ + 旋轉）
│   ├── gripper_pos.npy   # 夾爪開合狀態
│   └── instruction.txt   # 文字指令
├── episode_1/
├── episode_2/
...
└── episode_181/
```

**每個 episode 是一次完整的操作示範**，包含：
- 📷 影像序列（相機拍攝的場景）
- 📍 機器人位置序列（手臂移動軌跡）
- 💬 文字指令（描述要做什麼）

---

## 🎯 這個系統如何工作

### 訓練階段（如果你選擇訓練）
```
輸入: [相機影像 + "把芒果移到左邊" + 當前機器人位置]
     ↓
   VLA 模型
     ↓
輸出: [接下來 64 步的動作序列]
```

### 推論階段
```
你的指令: "把芒果往前推"
     ↓
   模型處理
     ↓
輸出: 機器人應該如何移動（位置、旋轉、夾爪）
```

### 動作定義（10維向量）
```python
action = [
    x, y, z,           # 位置（3維）
    r1, r2, r3,        # 旋轉（6維，6D表示）
    r4, r5, r6,
    gripper_open       # 夾爪（1維，0=閉合, 1=張開）
]
```

---

## 🎨 視覺化輸出範例

運行視覺化腳本後，你會得到：

### 1. 軌跡分析圖
- 3D 軌跡（鳥瞰圖）
- XY / XZ 平面投影
- 位置隨時間變化
- 速度分析
- 夾爪狀態變化

### 2. 動畫影片（GIF）
- 左側：相機視角
- 右側：機器人軌跡動畫
- 實時顯示當前位置和夾爪狀態

---

## 💡 常見場景

### 場景 1: 我只想看看資料長什麼樣
```bash
python simple_visualize_data.py --num_episodes 3
```

### 場景 2: 我想測試不同的指令
```bash
python interactive_demo.py
```

### 場景 3: 我想訓練一個模型
```bash
# 快速測試（10-30分鐘）
bash quick_train_demo.sh

# 完整訓練（數小時）
bash finetune.sh
```

### 場景 4: 我想看模型的預測效果
```bash
python visualize_inference.py \
    --checkpoint ./checkpoints/mango/xxx.ckpt \
    --num_episodes 5
```

---

## 🔧 安裝依賴（如果缺少）

```bash
# 基礎套件
pip install torch torchvision matplotlib opencv-python tqdm pillow numpy

# 訓練用（如果要訓練模型）
pip install transformers accelerate wandb

# 如果要用 Lightning
pip install pytorch-lightning
```

---

## 📖 檔案用途速查

| 檔案 | 用途 | 需要模型？ | 時間 |
|------|------|-----------|------|
| `simple_visualize_data.py` | 視覺化資料集 | ❌ 否 | 1-2分鐘 |
| `interactive_demo.py` | 互動式測試 | ❌ 否 | 即時 |
| `visualize_inference.py` | 視覺化模型預測 | ✅ 是 | 2-5分鐘 |
| `quick_train_demo.sh` | 快速訓練 | - | 10-30分鐘 |
| `finetune.sh` | 完整訓練 | - | 數小時 |

---

## ❓ 常見問題

### Q: 我現在應該做什麼？
**A:** 推薦順序：
1. 先運行 `simple_visualize_data.py` 看看資料
2. 再玩玩 `interactive_demo.py` 體驗指令規劃
3. 如果需要，再考慮訓練模型

### Q: 不訓練模型可以用嗎？
**A:** 可以！
- `simple_visualize_data.py` - 看真實資料
- `interactive_demo.py` - 用規則生成示範軌跡
- 如果有現有的 checkpoint，也可以直接用

### Q: 如何知道模型訓練得好不好？
**A:** 查看：
1. 訓練 loss 是否下降
2. 用 `visualize_inference.py` 比較預測 vs 真實軌跡
3. 位置誤差是否在可接受範圍（通常 < 0.05m）

### Q: 如何部署到實體機器人？
**A:** 目前腳本專注於視覺化。部署需要：
1. ROS2 介面（參考 `inference.sh` 中的 ros2 腳本）
2. 機器人控制器設定
3. 實時控制迴路
**建議**：先在模擬環境充分測試

### Q: 可以用自己的資料嗎？
**A:** 可以！資料格式要求：
```
your_dataset/
└── episode_X/
    ├── camera1/*.png        # 影像
    ├── ee_poses.npy         # (N, 7) [x,y,z, qx,qy,qz,qw]
    ├── gripper_pos.npy      # (N,)
    └── instruction.txt      # 文字指令
```

---

## 🎯 下一步建議

### 立即可做：
1. ✅ `python simple_visualize_data.py` - 看資料
2. ✅ `python interactive_demo.py` - 測試指令

### 短期（如果需要）：
3. 🔄 `bash quick_train_demo.sh` - 快速訓練
4. 🔄 收集更多示範資料

### 長期：
5. 🔄 完整訓練更大的模型
6. 🔄 部署到實體機器人
7. 🔄 整合觸覺感測

---

## 📞 需要幫助？

檢查清單：
- [ ] 確認資料集存在：`ls data/datasets/mango/`
- [ ] 確認 Python 環境：`python --version`（需要 3.8+）
- [ ] 安裝依賴：`pip install matplotlib opencv-python numpy tqdm`
- [ ] 查看文檔：`quick_start_zh.md`

---

**祝你順利！有任何問題隨時問 🎉**

---

## 📝 命令速查表

```bash
# 1. 視覺化資料（最簡單）
python simple_visualize_data.py

# 2. 互動式 demo
python interactive_demo.py

# 3. 快速訓練
bash quick_train_demo.sh

# 4. 視覺化模型結果
python visualize_inference.py --checkpoint ./outputs/demo_quick/last.ckpt

# 5. 檢查資料集
ls -l data/datasets/mango/ | head -20

# 6. 查看一個 episode 的內容
ls -la data/datasets/mango/episode_0/
```
