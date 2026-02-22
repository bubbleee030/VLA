# VLA Demo 資料夾

這個資料夾包含所有 VLA 機器人控制系統的 demo 相關檔案。

## 📁 資料夾結構

```
demo/
├── README.md                    # 本檔案
│
├── 📖 文檔（推薦閱讀順序）
│   ├── 一頁總結.md              # ⭐ 最快速的概覽（3 分鐘）
│   ├── START_HERE.md            # ⭐ 快速開始指南（5 分鐘）
│   ├── quick_start_zh.md        # 詳細中文教學
│   ├── README_DEMO.md           # 完整技術文檔
│   ├── simple_demo_guide.md     # 英文指南
│   └── FILES_SUMMARY.txt        # 所有檔案說明
│
├── 🔧 執行腳本
│   ├── install_demo_deps.sh     # 安裝依賴（第一次需要）
│   ├── run_demo.sh              # 互動式選單（主要入口）
│   └── quick_train_demo.sh      # 快速訓練腳本
│
└── 🐍 Python 程式
    ├── simple_visualize_data.py # 視覺化資料集（不需訓練）
    ├── interactive_demo.py      # 互動式指令測試（不需訓練）
    └── visualize_inference.py   # 視覺化模型推論結果
```

## 🚀 快速開始

### 方式 1: 互動式選單（推薦）

```bash
cd /home/cmwang16/VLA/demo
bash run_demo.sh
```

### 方式 2: 直接執行

```bash
cd /home/cmwang16/VLA/demo

# 第一次使用：安裝依賴
bash install_demo_deps.sh

# 視覺化資料集（不需訓練）
python3 simple_visualize_data.py --num_episodes 5

# 互動式指令測試（不需訓練）
python3 interactive_demo.py

# 訓練模型（可選）
bash quick_train_demo.sh

# 視覺化模型結果（訓練後）
python3 visualize_inference.py --checkpoint ../outputs/demo_quick/last.ckpt
```

## 📖 建議閱讀順序

1. **[一頁總結.md](一頁總結.md)** - 快速瀏覽整個系統（3 分鐘）
2. **[START_HERE.md](START_HERE.md)** - 詳細的快速開始指南（5 分鐘）
3. 實際運行 demo
4. **[quick_start_zh.md](quick_start_zh.md)** - 深入了解（需要時查閱）

## 🎯 各檔案用途

### 文檔檔案

| 檔案 | 用途 | 適合對象 |
|------|------|----------|
| `一頁總結.md` | 最精簡的總覽 | 想快速了解的人 |
| `START_HERE.md` | 快速開始指南 | 第一次使用 |
| `quick_start_zh.md` | 詳細中文教學 | 需要深入了解 |
| `README_DEMO.md` | 完整技術文檔 | 開發者 |
| `simple_demo_guide.md` | 英文版指南 | 英文使用者 |
| `FILES_SUMMARY.txt` | 所有檔案說明 | 快速查詢 |

### 執行腳本

| 檔案 | 功能 | 時間 |
|------|------|------|
| `install_demo_deps.sh` | 安裝必要的 Python 套件 | 1-5 分鐘 |
| `run_demo.sh` | 互動式選單，選擇要執行的功能 | - |
| `quick_train_demo.sh` | 快速訓練（1000 steps） | 10-30 分鐘 |

### Python 程式

| 檔案 | 功能 | 需要模型？ | 輸出位置 |
|------|------|-----------|----------|
| `simple_visualize_data.py` | 視覺化資料集 | ❌ | `../data_visualization/` |
| `interactive_demo.py` | 互動式指令測試 | ❌ | `../demo_outputs/interactive/` |
| `visualize_inference.py` | 視覺化模型推論 | ✅ | `../demo_outputs/` |

## 💡 使用提示

### 第一次使用

```bash
cd /home/cmwang16/VLA/demo

# 1. 安裝依賴
bash install_demo_deps.sh

# 2. 視覺化資料（了解資料集）
python3 simple_visualize_data.py --num_episodes 3

# 3. 測試指令（體驗互動）
python3 interactive_demo.py
```

### 訓練模型後

```bash
cd /home/cmwang16/VLA/demo

# 視覺化模型預測結果
python3 visualize_inference.py \
    --checkpoint ../outputs/demo_quick/last.ckpt \
    --num_episodes 5
```

## 📊 輸出位置

所有視覺化結果會儲存在 VLA 主目錄下：

```
/home/cmwang16/VLA/
├── demo/                        # 這個資料夾
├── data_visualization/          # 資料集視覺化結果
├── demo_outputs/                # 模型推論視覺化結果
│   └── interactive/             # 互動式 demo 結果
└── outputs/                     # 訓練的 checkpoints
    └── demo_quick/              # 快速訓練的輸出
```

## ❓ 常見問題

### Q: 我應該從哪裡開始？
**A:** 閱讀 [一頁總結.md](一頁總結.md)，然後執行 `bash run_demo.sh`

### Q: 必須訓練模型嗎？
**A:** 不用！前兩個 Python 腳本就能看到效果（視覺化和互動測試）

### Q: 資料集在哪裡？
**A:** `/home/cmwang16/VLA/data/datasets/mango/`（VLA 主目錄下）

### Q: 如何回到 VLA 主目錄？
**A:** `cd ..` 或 `cd /home/cmwang16/VLA`

## 🔗 相關連結

- VLA 主目錄：`/home/cmwang16/VLA`
- 資料集位置：`/home/cmwang16/VLA/data/datasets/mango/`
- 原始訓練腳本：`/home/cmwang16/VLA/finetune.sh`
- 主程式：`/home/cmwang16/VLA/main.py`

---

**準備好了嗎？開始吧！** 🚀

```bash
cd /home/cmwang16/VLA/demo
bash run_demo.sh
```
