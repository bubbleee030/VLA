# 快速開始指南 - VLA 機器人控制 Demo

## 📋 你的需求
✅ 下指令讓機器手臂移動到目標位置  
✅ 先碰觸物體就好，不需要夾取  
✅ 先做視覺化，不需要實體機器人  
✅ 使用現有的資料集進行 demo  

## 🎯 解決方案

你已經有 **mango 資料集**（181 個示範 episodes），可以直接用來訓練和測試！

### 資料集內容
每個 episode 包含：
- 📷 相機影像（camera1, camera2）
- 🖐️ 觸覺影像（gelsight）
- 📍 機器手臂位置和旋轉（ee_poses.npy）
- ✊ 夾爪開合狀態（gripper_pos.npy）
- 💬 文字指令（instruction.txt）

## 🚀 三步驟開始

### 步驟 1: 快速測試（不需要訓練）

直接測試現有的視覺化功能：

```bash
cd /home/cmwang16/VLA

# 互動式 demo - 輸入指令看規劃的軌跡
python interactive_demo.py
```

這會：
- ✅ 讓你輸入中文指令（例如："把芒果移到左邊"）
- ✅ 顯示 3D 軌跡規劃視覺化
- ✅ 不需要訓練模型，用規則生成示範軌跡

**範例指令：**
- "把芒果移到左邊"
- "抓住芒果並抬高"
- "把芒果往前推"

### 步驟 2: 視覺化現有資料

看看資料集裡實際的機器人動作：

```bash
# 安裝可能缺少的依賴
pip install matplotlib opencv-python tqdm

# 視覺化現有資料（不需要模型）
python visualize_inference.py \
    --checkpoint dummy \
    --dataset_path ./data/datasets/mango \
    --num_episodes 5 \
    --output_dir ./demo_outputs
```

這會生成：
- 📊 3D 軌跡圖
- 🎬 動畫影片（GIF）
- 📈 位置誤差分析

結果會儲存在 `./demo_outputs/` 資料夾。

### 步驟 3: 訓練模型（可選）

如果想訓練自己的模型：

```bash
# 快速訓練（1000 steps，約 10-30 分鐘）
bash quick_train_demo.sh
```

訓練完成後，可以用訓練好的模型做推論：

```bash
python visualize_inference.py \
    --checkpoint ./outputs/demo_quick/last.ckpt \
    --dataset_path ./data/datasets/mango \
    --num_episodes 5
```

## 📊 訓練資料說明

### 輸入
- **影像**: 相機拍攝的場景
- **指令**: 文字描述（例如："把芒果移到左邊"）
- **當前狀態**: 機器手臂目前的位置和夾爪狀態

### 輸出
- **動作序列**: 接下來 64 步的動作
  - 位置 (x, y, z)
  - 旋轉（6D 表示）
  - 夾爪開合 (0=閉合, 1=張開)

### 模型如何學習
```
訓練時：
  輸入 → [影像 + "把芒果移到左邊" + 當前位置]
  目標 → [人類示範的 64 步動作]
  
推論時：
  輸入 → [影像 + 你的指令 + 當前位置]
  輸出 → [模型預測的 64 步動作]
```

## 🎨 視覺化內容

執行 demo 後你會看到：

1. **3D 軌跡圖**
   - 藍線：真實軌跡（人類示範）
   - 紅線：預測軌跡（模型生成）
   - 綠點：起點
   - 紅星：終點

2. **XY 平面投影**
   - 從上方俯視的軌跡

3. **位置誤差曲線**
   - 預測與真實的差距隨時間變化

4. **動畫影片**
   - 左側：相機影像
   - 右側：機器人移動軌跡

## 📝 常見問題

### Q1: 我可以用自己的指令嗎？
**A:** 可以！有兩種方式：
1. **不訓練**：使用 `interactive_demo.py`，輸入任何中文指令，會用規則生成軌跡
2. **訓練後**：模型會學習理解類似的指令，可以泛化到新指令

### Q2: 如何收集自己的資料？
**A:** 需要記錄：
```
episode_0/
  ├── camera1/          # 相機影像（PNG）
  │   ├── 000000.png
  │   ├── 000001.png
  │   └── ...
  ├── ee_poses.npy      # 末端執行器位置 (N, 7) [x,y,z,qx,qy,qz,qw]
  ├── gripper_pos.npy   # 夾爪狀態 (N,)
  └── instruction.txt   # 文字指令
```

### Q3: 不想訓練，可以直接用嗎？
**A:** 可以！兩種方式：
1. 使用 `interactive_demo.py` - 規則生成（立即可用）
2. 使用現有的 checkpoint（如果有的話）

### Q4: 訓練需要多久？
- **快速測試** (1000 steps): 10-30 分鐘
- **完整訓練** (40000 steps): 數小時到1天（取決於GPU）

### Q5: 需要什麼硬體？
- **最低**：CPU + 8GB RAM（會很慢）
- **建議**：NVIDIA GPU (8GB+ VRAM)
- **最佳**：NVIDIA GPU (24GB+ VRAM)

## 🔧 故障排除

### 錯誤：找不到資料集
```bash
# 確認資料集位置
ls -la data/datasets/mango/
```

### 錯誤：缺少模組
```bash
# 安裝依賴
pip install torch torchvision matplotlib opencv-python tqdm pillow
pip install transformers accelerate
```

### 錯誤：CUDA out of memory
```bash
# 減小 batch size
# 編輯 quick_train_demo.sh:
# --train_batch_size=1  # 原本是 2
```

## 📚 下一步

1. ✅ **現在就試試**: `python interactive_demo.py`
2. 📊 **看看資料**: `python visualize_inference.py --dataset_path ./data/datasets/mango --num_episodes 3`
3. 🎓 **訓練模型**: `bash quick_train_demo.sh`
4. 🤖 **部署到實體機器人**（之後）

## 📖 檔案說明

| 檔案 | 用途 |
|------|------|
| `simple_demo_guide.md` | 完整文檔（英文） |
| `quick_start_zh.md` | 本檔案（中文快速開始） |
| `quick_train_demo.sh` | 快速訓練腳本 |
| `interactive_demo.py` | 互動式指令測試 |
| `visualize_inference.py` | 視覺化資料集和模型預測 |
| `finetune.sh` | 完整訓練腳本 |
| `main.py` | 訓練主程式 |

## 💡 提示

- 先用 `interactive_demo.py` 快速體驗
- 再用 `visualize_inference.py` 看真實資料
- 最後才考慮訓練模型
- 訓練可以先跑 1000 steps 測試流程

---

**需要幫助？** 檢查：
- 資料集是否存在：`data/datasets/mango/`
- Python 環境是否正確
- 必要的套件是否安裝

祝你順利！🎉
