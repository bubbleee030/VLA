# 🤖 VLA - 視覺語言行動機器人學習系統

<div align="center">

**一個用於機器人學習和控制的視覺-語言-行動整合框架**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

</div>

---

## 📌 項目介紹

VLA 是一個先進的機器人學習系統，集成了**視覺感知**、**語言理解**和**控制行動**三個核心模塊。系統能夠理解自然語言指令，並根據視覺輸入生成相應的機器人控制命令。

### 🎯 核心特性

- ✅ **多模態學習** - 融合視覺和語言信息
- ✅ **自然語言指令** - 支持中英文自然語言命令
- ✅ **觸覺反饋集成** - 包含觸覺傳感器数據处理（可选）
- ✅ **模塊化設計** - 易於擴展和自定義
- ✅ **完整 Demo** - 提供開箱即用的演示代碼
- ✅ **詳細文檔** - 中文教學和技術文檔

---

## 📁 項目結構

```
VLA/
├── models/                 # 核心模型架構
│   ├── rdt/               # RDT (Robotics Diffusion Transformer) 主模型
│   ├── multimodal_encoder/ # 多模態編碼器
│   └── rdt_runner.py      # 模型推論入口
│
├── train/                 # 訓練代碼
│   ├── train.py          # 主訓練迴圈
│   └── *.py              # 訓練工具函數
│
├── data/                  # 數據處理模塊
│   ├── hdf5_vla_dataset.py       # HDF5 數據加載
│   ├── unified_vla_dataset.py    # 統一數據格式
│   └── create_controller_dataset_episode.py # 數據轉換
│
├── configs/               # 配置文件
│   ├── base.yaml         # 基礎配置
│   ├── train_config.yaml # 訓練配置
│   └── *.json            # 數據集和權重配置
│
├── demo/                  # 演示和快速開始
│   ├── run_demo.sh       # 互動式 demo 菜單
│   ├── simple_visualize_data.py  # 數據可視化
│   ├── interactive_demo.py       # 互動推論測試
│   ├── START_HERE.md     # 快速開始指南
│   └── 一頁總結.md       # 項目總覽
│
├── scripts/               # 實用腳本
├── main.py               # 直接訓練入口
├── finetune.sh          # 微調腳本
└── inference.sh         # 推論腳本
```

---

## 🚀 快速開始

### 💻 環境設置

```bash
# 克隆項目
git clone https://github.com/your-username/VLA.git
cd VLA

# 創建虛擬環境（建議使用 conda）
conda create -n vla python=3.9
conda activate vla

# 安裝依賴
pip install -r requirements.txt
```

### 🎮 運行 Demo（推薦首先嘗試）

```bash
cd demo

# 安裝 demo 依賴（首次需要）
bash install_demo_deps.sh

# 執行互動式菜單（最簡單的方式）
bash run_demo.sh
```

**Demo 菜單包括：**
- 📊 視覺化數據集（無需訓練）
- 🎯 互動推論測試（快速驗證模型）
- 📈 訓練小規模模型（可選）

### 📖 推薦閱讀順序

1. **[`demo/一頁總結.md`](demo/一頁總結.md)** - 3 分鐘快速了解（⭐ 強烈推薦）
2. **[`demo/START_HERE.md`](demo/START_HERE.md)** - 5 分鐘詳細入門
3. 實際運行 demo 代碼
4. **[`demo/quick_start_zh.md`](demo/quick_start_zh.md)** - 進階設置（需要時查閱）

---

## 🔧 主要訓練命令

### 訓練完整模型

```bash
python main.py \
    --config_path configs/train_config.yaml \
    --output_dir checkpoints/my_model \
    --load_from_hdf5
```

### 微調預訓練模型

```bash
bash finetune.sh
```

### 推論和評估

```bash
bash inference.sh
```

詳細參數說明見 [`main.py`](main.py) 和 [配置文件](configs/train_config.yaml)

---

## 📊 主要組件說明

### 🧠 模型架構

| 組件 | 功能 | 說明 |
|------|------|------|
| **RDT Model** | 主模型框架 | Robotics Diffusion Transformer |
| **Vision Encoder** | 圖像編碼 | 處理視覺輸入 |
| **Language Encoder** | 文本編碼 | 理解自然語言指令 |
| **Action Decoder** | 行動生成 | 輸出機器人控制信號 |

### 📚 數據格式

支持多種數據格式：
- **HDF5** - 高效儲存和加載
- **TFRecord** - TensorFlow 標準格式
- **統一格式** - 自定義數據集支持

### 🎛️ 可配置項

所有主要參數在 [`configs/`](configs/) 中配置：
- 模型大小和架構
- 訓練超參數
- 數據集位置和權重
- 推論參數

---

## 🖼️ 功能演示

### 同時支持的功能

```
🎥 視覺輸入  →  🧠 多模態編碼  →  🎯 行動預測  →  🤖 機器人執行
                      ↓
                 自然語言指令
```

### 使用案例

- ✋ **物體抓取** - 理解"拿起紅色方塊"並執行
- 🔄 **軌跡複製** - 從視覺觀察學習行為
- 🎓 **遷移學習** - 在新任務上快速微調
- 💡 **多機器人** - 支持多種機器人平台

---

## 📈 項目亮點

### 相比基線的改進

- 📊 **多模態融合** - 結合視覺和語言的雙重優勢
- 🎯 **精度提升** - [具體數字待填充]
- ⚡ **計算效率** - [具體數字待填充]
- 🔧 **易用性** - 開箱即用的 demo 和文檔

### 技術選型

- **模型基礎** - Diffusion Transformers
- **訓練框架** - PyTorch + Accelerate
- **多機訓練** - DeepSpeed 支持
- **機器人控制** - [支持的平台]

---

## 📋 系統要求

```
Python >= 3.8
PyTorch >= 1.13
CUDA 11.8+ (推薦使用 GPU)
RAM >= 16GB (訓練建議 32GB+)
GPU VRAM >= 24GB (用於完整訓練)
```

完整依賴見 [`requirements.txt`](requirements.txt)

---

## 🤝 如何使用（履歷用途）

### 對於面試和項目展示

1. **快速演示**（5 分鐘）
   - 克隆 repo
   - 運行 `bash demo/run_demo.sh`
   - 展示可視化效果

2. **代碼審查**（深度技術討論）
   - 核心模型在 [`models/rdt/`](models/rdt/)
   - 訓練邏輯在 [`train/`](train/)
   - 數據處理在 [`data/`](data/)

3. **性能展示**（技術亮點）
   - 查看 [`test_results.txt`](test_results.txt) 的評估指標
   - 詢問具體的改進點和技術細節

---

## 🔍 文件導航

### 快速查找

- 📖 **想快速了解？** → [`demo/一頁總結.md`](demo/一頁總結.md)
- 🚀 **想立即開始？** → [`demo/START_HERE.md`](demo/START_HERE.md)
- 📚 **想深入學習？** → [`demo/quick_start_zh.md`](demo/quick_start_zh.md)
- 💻 **想修改代碼？** → [`models/`](models/) 和 [`train/`](train/)
- ⚙️ **想調整參數？** → [`configs/`](configs/)

---

## 📞 技術支持

### 常見問題

**Q: 沒有 GPU 能運行嗎？**  
A: 可以，但很慢。建議使用 demo 中的數據可視化來理解系統。

**Q: 能用自己的數據嗎？**  
A: 可以，見 `demo/prepare_own_dataset.py`

**Q: 支持哪些機器人？**  
A: 主要是 Franka，可擴展支持其他平台

---

## 📄 許可證

MIT License - 詳見 [LICENSE](LICENSE)

---

## 🙏 致謝

本項目基於以下研究和框架：
- Robotics Diffusion Transformer
- Vision-Language 多模態學習
- [相關論文和項目]

---

<div align="center">

**⭐ 如果這個項目對你有幫助，請給個 Star！**

Made with ❤️ for Robotics & AI

</div>
