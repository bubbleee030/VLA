# VLA - Vision Language Action Robotics Learning System

<div align="center">

**A vision-language-action integrated framework for robot learning and control**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

</div>

---

## Overview

VLA is an advanced robotics learning system that integrates **visual perception**, **language understanding**, and **action control** as three core modules. The system can understand natural language instructions and generate corresponding robot control commands based on visual input.

### Key Features

- Multi-modal Learning - Integrate vision and language information
- Natural Language Instructions - Support Chinese and English commands
- Tactile Feedback Integration - Include tactile sensor data processing (optional)
- Modular Design - Easy to extend and customize
- Complete Demo - Provide out-of-the-box demonstration code
- Detailed Documentation - Chinese tutorials and technical documentation

---

## Project Structure

```
VLA/
├── models/                 # Core model architecture
│   ├── rdt/               # RDT (Robotics Diffusion Transformer) main model
│   ├── multimodal_encoder/ # Multi-modal encoder
│   └── rdt_runner.py      # Model inference entry point
│
├── train/                 # Training code
│   ├── train.py          # Main training loop
│   └── *.py              # Training utility functions
│
├── data/                  # Data processing module
│   ├── hdf5_vla_dataset.py       # HDF5 data loading
│   ├── unified_vla_dataset.py    # Unified data format
│   └── create_controller_dataset_episode.py # Data conversion
│
├── configs/               # Configuration files
│   ├── base.yaml         # Base configuration
│   ├── train_config.yaml # Training configuration
│   └── *.json            # Dataset and weight configuration
│
├── demo/                  # Demo and quick start
│   ├── run_demo.sh       # Interactive demo menu
│   ├── simple_visualize_data.py  # Data visualization
│   ├── interactive_demo.py       # Interactive inference testing
│   ├── START_HERE.md     # Quick start guide
│   └── 一頁總結.md       # Project overview
│
├── scripts/               # Utility scripts
├── main.py               # Direct training entry
├── finetune.sh          # Fine-tuning script
└── inference.sh         # Inference script
```

---

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/VLA.git
cd VLA

# Create virtual environment (recommended with conda)
conda create -n vla python=3.9
conda activate vla

# Install dependencies
pip install -r requirements.txt
```

### Running Demo (Recommended First)

```bash
cd demo

# Install demo dependencies (first time only)
bash install_demo_deps.sh

# Execute interactive menu (simplest way)
bash run_demo.sh
```

**Demo menu includes:**
- Data visualization (no training required)
- Interactive inference testing (quick model verification)
- Small-scale model training (optional)

### Recommended Reading Order

1. **[`demo/一頁總結.md`](demo/一頁總結.md)** - Quick overview in 3 minutes (highly recommended)
2. **[`demo/START_HERE.md`](demo/START_HERE.md)** - Detailed introduction in 5 minutes
3. Run the demo code in practice
4. **[`demo/quick_start_zh.md`](demo/quick_start_zh.md)** - Advanced setup (consult when needed)

---

## Main Training Commands

### Train Full Model

```bash
python main.py \
    --config_path configs/train_config.yaml \
    --output_dir checkpoints/my_model \
    --load_from_hdf5
```

### Fine-tune Pretrained Model

```bash
bash finetune.sh
```

### Inference and Evaluation

```bash
bash inference.sh
```

See [`main.py`](main.py) and [configuration files](configs/train_config.yaml) for detailed parameter descriptions

---

## Core Components Overview

### Model Architecture

| Component | Function | Description |
|------|------|------|
| **RDT Model** | Main model framework | Robotics Diffusion Transformer |
| **Vision Encoder** | Image encoding | Process visual input |
| **Language Encoder** | Text encoding | Understand natural language instructions |
| **Action Decoder** | Action generation | Output robot control signals |

### Data Format

Support multiple data formats:
- **HDF5** - Efficient storage and loading
- **TFRecord** - TensorFlow standard format
- **Unified format** - Custom dataset support

### Configurable Items

All main parameters are configured in [`configs/`](configs/):
- Model size and architecture
- Training hyperparameters
- Dataset location and weights
- Inference parameters

---

## Functional Demonstration

### Supported Functionality

```
Visual Input  →  Multi-modal Encoding  →  Action Prediction  →  Robot Execution
                      ↓
                Natural Language Instructions
```

### Use Cases

- **Object Grasping** - Understand "pick up red cube" and execute
- **Trajectory Imitation** - Learn behavior from visual observation
- **Transfer Learning** - Quick fine-tuning on new tasks
- **Multi-Robot** - Support multiple robot platforms

---

## Project Highlights

### Improvements over Baselines

- **Multi-modal Fusion** - Combine the dual advantages of vision and language
- **Accuracy Improvement** - [Specific numbers to be filled]
- **Computational Efficiency** - [Specific numbers to be filled]
- **Usability** - Out-of-the-box demo and documentation

### Technical Stack

- **Model Foundation** - Diffusion Transformers
- **Training Framework** - PyTorch + Accelerate
- **Multi-machine Training** - DeepSpeed support
- **Robot Control** - [Supported platforms]

---

## System Requirements

```
Python >= 3.8
PyTorch >= 1.13
CUDA 11.8+ (GPU recommended)
RAM >= 16GB (32GB+ recommended for training)
GPU VRAM >= 24GB (for full training)
```

See [`requirements.txt`](requirements.txt) for complete dependencies

---

## How to Use (For Resume)

### For Interview and Project Presentations

1. **Quick Demo** (5 minutes)
   - Clone the repository
   - Run `bash demo/run_demo.sh`
   - Display visualization results

2. **Code Review** (In-depth technical discussion)
   - Core model in [`models/rdt/`](models/rdt/)
   - Training logic in [`train/`](train/)
   - Data processing in [`data/`](data/)

3. **Performance Showcase** (Technical highlights)
   - Check evaluation metrics in [`test_results.txt`](test_results.txt)
   - Discuss specific improvements and technical details

---

## File Navigation

### Quick Reference

- **Want a quick understanding?** → [`demo/一頁總結.md`](demo/一頁總結.md)
- **Want to start immediately?** → [`demo/START_HERE.md`](demo/START_HERE.md)
- **Want to learn in depth?** → [`demo/quick_start_zh.md`](demo/quick_start_zh.md)
- **Want to modify code?** → [`models/`](models/) and [`train/`](train/)
- **Want to adjust parameters?** → [`configs/`](configs/)

---

## Technical Support

### FAQ

**Q: Can it run without GPU?**  
A: Yes, but slowly. It is recommended to use data visualization in the demo to understand the system.

**Q: Can I use my own data?**  
A: Yes, see `demo/prepare_own_dataset.py`

**Q: Which robots are supported?**  
A: Primarily Franka, with extensibility to support other platforms

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Acknowledgments

This project is based on the following research and frameworks:
- Robotics Diffusion Transformer
- Vision-Language Multi-modal Learning
- VLA-Touch

---

<div align="center">

If this project is helpful to you, please give it a Star!

Made for Robotics & AI

</div>
