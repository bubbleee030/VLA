# 🚀 立即開始 - VLA Demo

1. ✅ **下指令** → 機器手臂移動到目標位置
2. ✅ **碰到物體就好** → 不需要抓取
3. ✅ **先視覺化** → 不需要實體機器人
4. ✅ **用現有資料** → 已有 mango 資料集（181 episodes）

---

## ⚡ 三步驟立即開始

### 步驟 1: 安裝依賴（第一次需要）

```bash
cd /home/cmwang16/VLA
bash install_demo_deps.sh
```

### 步驟 2: 運行 Demo

```bash
bash run_demo.sh
```

會出現選單：
```
1) 視覺化資料集（最簡單，不需訓練）  ← 推薦先選這個
2) 互動式指令測試
3) 快速訓練模型
4) 視覺化模型推論結果
5) 檢查資料集狀態
```

### 步驟 3: 查看結果

結果會儲存在：
- `./data_visualization/` - 資料集視覺化
- `./demo_outputs/` - 模型推論結果

---

## 🎯 推薦流程

### 第一次使用（不需訓練）

```bash
cd /home/cmwang16/VLA

# 1. 安裝依賴
bash install_demo_deps.sh

# 2. 視覺化資料（看看資料長什麼樣）
python3 simple_visualize_data.py --num_episodes 5

# 3. 互動式測試（輸入指令看規劃）
python3 interactive_demo.py
```

**總時間：約 5 分鐘**

### 如果想訓練模型

```bash
# 快速訓練（1000 steps，10-30分鐘）
bash quick_train_demo.sh

# 訓練完成後，視覺化結果
python3 visualize_inference.py \
    --checkpoint ./outputs/demo_quick/last.ckpt \
    --num_episodes 5
```

---

## 📊 你會看到什麼？

### 視覺化資料集
- **3D 軌跡圖**：機器人手臂的移動路徑
- **平面投影**：從不同角度看軌跡
- **速度分析**：移動速度隨時間變化
- **夾爪狀態**：開合狀態時序圖
- **動畫影片**：相機視角 + 軌跡動畫

### 互動式 Demo
- 輸入：`"把芒果移到左邊"`
- 輸出：3D 軌跡規劃圖，顯示手臂應該如何移動

---

## 🎨 範例指令

可以在互動式 Demo 中測試：

```
✅ "把芒果移到左邊"
✅ "把芒果移到右邊"
✅ "把芒果往前推"
✅ "抓住芒果並抬高"
✅ "輕輕碰觸芒果"
✅ "把芒果放下"
```

---

## 📁 資料集說明

你的 `mango` 資料集包含：

```
data/datasets/mango/
├── episode_0/              ← 一次完整的操作示範
│   ├── camera1/*.png       ← 相機影像序列
│   ├── camera2/*.png       ← 另一個視角
│   ├── gelsight/*.png      ← 觸覺感測影像
│   ├── ee_poses.npy        ← 手臂位置 (N, 7) [x,y,z, 四元數]
│   ├── gripper_pos.npy     ← 夾爪狀態 (N,)
│   └── instruction.txt     ← 文字指令
├── episode_1/
├── episode_2/
...
└── episode_181/
```

**總共 181 個示範** - 足夠訓練一個基礎模型！

---

## 💡 重要概念

### 什麼是 Episode？
一個 episode = 一次完整的操作示範（從開始到結束）

例如：
- 人類示範："把芒果移到左邊"
- 記錄：每一幀的影像 + 手臂位置 + 夾爪狀態
- 儲存：episode_X/

### 模型如何學習？

```
訓練時：
  輸入: [相機影像, "把芒果移到左邊", 當前手臂位置]
  目標: [人類示範的 64 步動作序列]
  
  模型學習：看到這個場景和指令時，應該如何移動

推論時：
  輸入: [新的影像, 你的指令, 當前位置]
  輸出: [模型預測的 64 步動作]
```

### 動作是什麼？

每一步的動作是 **10 維向量**：
```python
[
  x, y, z,              # 位置（3維）
  r1, r2, r3,           # 旋轉（6維，6D表示）
  r4, r5, r6,
  gripper_open          # 夾爪（0=閉合, 1=張開）
]
```

---

## ❓ 常見問題

### Q: 我需要訓練模型嗎？
**A:** 不一定！你可以：
- ✅ 先用視覺化工具看資料
- ✅ 用互動式 Demo 測試指令（不需訓練）
- 🔄 需要實際預測時才訓練

### Q: 訓練需要多久？
**A:**
- 快速測試（1000 steps）：10-30 分鐘
- 完整訓練（40000 steps）：數小時
- 取決於：GPU 性能、資料量

### Q: 我沒有 GPU 可以嗎？
**A:** 可以！但是：
- 視覺化工具：CPU 就夠（很快）
- 訓練模型：CPU 會很慢（建議用 GPU）

### Q: 如何部署到實體機器人？
**A:** 
1. 先在視覺化環境充分測試
2. 使用 ROS2 介面（repo 中有範例）
3. 需要實體機器人的控制介面
4. 建議：先完成步驟 1-3

### Q: 可以改指令嗎？
**A:** 可以！
- 互動式 Demo：輸入任何中文指令
- 訓練模型：會學習理解類似的指令
- 收集新資料：可以用新指令

---

## 🔧 故障排除

### 找不到 python
```bash
# 使用 python3
python3 --version

# 或建立連結
sudo apt install python-is-python3
```

### 缺少模組
```bash
# 重新安裝依賴
bash install_demo_deps.sh

# 或手動安裝
pip3 install numpy matplotlib opencv-python pillow tqdm torch
```

### 找不到資料集
```bash
# 檢查路徑
ls -la /home/cmwang16/VLA/data/datasets/mango/

# 如果在其他位置，使用 --dataset_path 參數
python3 simple_visualize_data.py --dataset_path /path/to/your/dataset
```

---

## 📚 文檔索引

| 檔案 | 內容 |
|------|------|
| **`START_HERE.md`** | **本檔案 - 快速開始** |
| `README_DEMO.md` | 完整說明文檔 |
| `quick_start_zh.md` | 詳細的中文教學 |
| `simple_demo_guide.md` | 英文版指南 |

---

## 🎯 下一步

### 現在就做：
```bash
# 1. 安裝依賴
bash install_demo_deps.sh

# 2. 看看資料
python3 simple_visualize_data.py --num_episodes 3

# 3. 測試指令
python3 interactive_demo.py
```

### 之後可以：
- 🔄 訓練自己的模型
- 🔄 收集更多資料
- 🔄 調整模型參數
- 🔄 部署到實體機器人

---

## ✅ 檢查清單

在開始前確認：
- [ ] Python 3.8+ 已安裝：`python3 --version`
- [ ] 資料集存在：`ls data/datasets/mango/`
- [ ] 依賴已安裝：`bash install_demo_deps.sh`
- [ ] 在正確目錄：`cd /home/cmwang16/VLA`

---

**準備好了嗎？開始吧！** 🚀

```bash
cd /home/cmwang16/VLA
bash run_demo.sh
```

有問題隨時問！ 😊
