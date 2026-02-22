# VLA 專案進度報告
## Vision-Language-Action Model for Robotic Manipulation

**報告人：[您的名字]**  
**日期：[日期]**

---

## 1. 專案目標

### 研究目標
實作視覺-語言-動作（VLA）模型，讓機器手臂能理解文字指令並執行相應動作

### 核心功能
- **輸入**：文字指令 + 相機影像（+ 觸覺資訊）
- **輸出**：機器手臂控制指令（位置 + 旋轉 + 夾爪）
- **應用**：物體抓取、操作、組裝等機器人任務

### 範例
```
指令：「把芒果移到左邊」
  ↓
[模型處理]
  ↓
輸出：機器手臂軌跡序列
```

---

## 2. 技術架構

### 模型：RDT (Robotics Diffusion Transformer)

```
┌──────────────┐
│ 文字指令     │ → T5-XXL (4.7B)
└──────────────┘
                              ┌─────────────────┐
┌──────────────┐              │                 │
│ 相機影像     │ → SigLIP    →│ Transformer     │→ 動作序列
└──────────────┘   (400M)     │ + Diffusion     │   [T, 128]
                              │                 │
┌──────────────┐              └─────────────────┘
│ 觸覺影像     │ → ResNet-18
└──────────────┘   (可選)
```

### 關鍵技術
- **多模態融合**：整合視覺、語言、觸覺資訊
- **擴散模型**：生成平滑的動作序列
- **Transformer**：捕捉長期依賴關係

**參數量**：約 1.2B（diffusion model）

---

## 3. 資料集

### 使用的資料集

| 資料集 | Episodes | 任務 | 用途 |
|--------|----------|------|------|
| **Mango** | 181 | 抓取芒果（左/右/削皮） | 預訓練/微調 |
| **Own** | 4 | 工具箱/大板子操作 | 微調/測試 |

### Episode 資料結構

```
episode_0/
├── camera1/              # 相機影像序列
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── ee_poses.npy          # 手臂位置與旋轉 [T, 7]
│                         # (x, y, z, qx, qy, qz, qw)
├── gripper_pos.npy       # 夾爪狀態 [T,]
└── metadata.json         # 文字指令等資訊
```

### 資料統計
- 平均 episode 長度：~80 步（8 秒 @ 10 Hz）
- 影像解析度：640×480 → 224×224
- 控制頻率：10 Hz
- 資料增強：隨機裁切、色彩抖動

---

## 4. 實作成果（1/3）- 工具開發

### 建立的工具（共 16 個檔案）

#### 📊 資料處理與視覺化
1. `prepare_own_dataset.py` - 資料格式轉換
2. `simple_visualize_data.py` - 3D 軌跡視覺化
3. `visualize_inference.py` - 模型預測視覺化

#### 🤖 Demo 與測試
4. `interactive_demo.py` - 互動式指令測試
5. `train_with_own_data.sh` - 自動化訓練流程

#### 🔧 訓練腳本
6. `quick_train_demo.sh` - 快速訓練（1000 steps）
7. `run_demo.sh` - 互動式選單

#### 📚 文檔
8-13. 完整的中英文使用指南

#### 🛠 技術實作
14. `DirectoryVLADataset` - 自定義資料載入器
15-16. 配置檔案與環境設定

**總開發量**：約 2000+ 行程式碼

---

## 4. 實作成果（2/3）- 資料視覺化

### 3D 軌跡分析

[插入圖片：episode_0_trajectory.png]

**包含內容：**
- 3D 軌跡路徑（起點→終點）
- XY、XZ 平面投影
- 位置-時間曲線（X, Y, Z）
- 速度-時間曲線
- 夾爪狀態

**關鍵發現：**
- ✓ 軌跡平滑連續
- ✓ 速度變化合理（無突變）
- ✓ 資料品質良好

---

## 4. 實作成果（3/3）- 訓練與部署

### 訓練配置

```yaml
模型：RDT (1.2B 參數)
資料集：Mango (181 episodes)
批次大小：2
學習率：1e-4
混合精度：BF16
訓練步數：1000 (測試) / 40000 (完整)

硬體環境：
- GPU：NVIDIA RTX 5090
- VRAM 使用：~24 GB
- 訓練時間：~20 分鐘 (1000 steps)
```

### 訓練進度

[如果有的話插入 loss curve]

```
Steps:   0%|          | 10/1000 [00:35<58:15]
Loss: 0.245 → 0.132 → 0.089 (持續下降中)
```

### 已完成
- ✓ 環境建置完成
- ✓ 資料載入成功
- ✓ 模型訓練啟動
- ✓ Loss 正常下降

---

## 5. 挑戰與解決方案

### 技術挑戰彙總

| # | 挑戰 | 原因 | 解決方案 | 學習重點 |
|---|------|------|----------|----------|
| 1 | 配置檔案缺失 | 跨 repo 依賴 | 複製 RDT repo 配置 | 依賴管理 |
| 2 | 路徑問題 | demo/ 相對路徑 | 統一使用 ../ | 專案結構 |
| 3 | 資料格式不符 | 目錄 vs HDF5 | 實作 DirectoryVLADataset | 自定義 Dataset |
| 4 | 張量維度不匹配 | state_dim 差異 | padding + fill_in_state | 張量操作 |
| 5 | DeepSpeed 衝突 | PyTorch 2.10 不兼容 | 升級到 0.18.3 | 版本管理 |
| 6 | 中文字體顯示 | matplotlib 預設 | NotoSansTC-Variable.ttf | 視覺化 |

### 核心收穫
- ✓ PyTorch/Transformers 生態系統
- ✓ 多模態模型架構理解
- ✓ 機器人資料處理流程
- ✓ Debugging 與問題排查能力

---

## 6. Demo 展示

### 展示 1：資料視覺化

**現場操作：**
```bash
cd /home/cmwang16/VLA/demo
python3 simple_visualize_data.py --num_episodes 5
```

**展示內容：**
- 3D 軌跡動畫
- 多視角投影圖
- 時序分析曲線
- 相機視角 + 軌跡動畫（GIF）

---

### 展示 2：互動式指令測試

**現場操作：**
```bash
python3 interactive_demo.py
```

**測試指令：**
```
輸入: "把芒果移到左邊"
輸出: [顯示規劃的軌跡]

輸入: "移動到工具箱"
輸出: [顯示規劃的軌跡]
```

---

### 展示 3：訓練監控（如果時間允許）

**現場操作：**
```bash
tail -f ../outputs/demo_quick/log.txt
```

或

```bash
tensorboard --logdir ../outputs/demo_quick/
```

---

## 7. 下一步計劃

### 短期目標（1-2 週）

#### ✓ 資料準備
- [ ] 將 arm_position.docx 轉換為 CSV
- [ ] 運行 `prepare_own_dataset.py`
- [ ] 視覺化檢查資料品質

#### ✓ 模型訓練
- [ ] 混合訓練（Mango + Own）
- [ ] 完整訓練 10000+ steps
- [ ] 評估與優化

#### ✓ 部署測試
- [ ] 整合 ROS 控制
- [ ] 實機測試
- [ ] 安全性驗證

---

### 中長期目標（1-2 個月）

#### 📊 資料擴展
- 收集更多場景資料（20+ episodes）
- 增加任務多樣性（抓取、放置、組裝）
- 加入觸覺資訊

#### 🔧 模型優化
- 超參數調整（學習率、批次大小）
- 架構改進（注意力機制、更大模型）
- 多任務學習

#### 🤖 實際應用
- 長期穩定性測試
- 多物體場景
- 動態環境適應

---

## 8. 總結

### 已完成工作

✅ **環境與工具**
- 完整的訓練與測試環境
- 13+ 個實用工具與腳本
- 完善的中英文文檔

✅ **資料處理**
- 資料視覺化工具
- 自動化資料轉換
- 資料品質驗證

✅ **模型訓練**
- 訓練管線建置
- 多資料集支援
- 監控與日誌系統

✅ **問題解決**
- 解決 6+ 個技術挑戰
- 實作自定義 Dataset
- 環境相容性修復

---

### 技術收穫

🎓 **知識掌握**
- 多模態深度學習（Vision + Language + Action）
- 擴散模型原理與應用
- Transformer 架構

🛠 **實戰能力**
- PyTorch 生態系統（Accelerate、DeepSpeed）
- 機器人資料處理
- 大規模模型訓練

🔍 **軟技能**
- 問題排查與 debugging
- 文檔閱讀與理解
- 工具開發與自動化

---

### 下一步行動

📅 **本週**
- 完成自己資料集的準備
- 開始混合訓練

📅 **下週**
- 完成完整訓練
- 準備實機部署

📅 **月底**
- 實機測試
- 收集反饋與改進

---

## 謝謝！

**問題與討論**

---

### 附錄：關鍵程式碼片段

#### DirectoryVLADataset 實作

```python
class DirectoryVLADataset:
    """載入目錄結構的 episode 資料"""
    
    def load_episode(self, episode_path):
        # 載入影像
        images = self.load_images(episode_path / "camera1")
        
        # 載入機器人狀態
        ee_poses = np.load(episode_path / "ee_poses.npy")
        gripper = np.load(episode_path / "gripper_pos.npy")
        
        # 轉換格式
        qpos = self.convert_to_qpos(ee_poses, gripper)
        
        # 填充到 state_dim
        state = self.fill_in_state(qpos, STATE_DIM)
        
        return {"images": images, "state": state, ...}
```

---

### 附錄：資源連結

- **專案 GitHub**：[RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer)
- **相關論文**：[VLA-Touch Paper](https://arxiv.org/...)
- **文檔位置**：`/home/cmwang16/VLA/demo/`

---
