# 🧠 Image Segmentation Assignment

This repository provides an end-to-end pipeline for semantic image segmentation using **PyTorch** and **PyTorch Lightning**. It includes data preprocessing from COCO annotations, model training using DeepLabV3+, and visual inference outputs. This project emphasizes reproducibility, robustness, and evaluation under computational constraints.

---

## 📌 Project Overview

- **Task 1 – Dataset Preparation**: `data_preprocessing.py` script processes COCO annotations and images to generate segmentation masks.
- **Task 2 – Model Training**: `train.py` implements a configurable PyTorch Lightning trainer with DeepLabV3+ backbone from `segmentation_models_pytorch`.

---

## 🧰 Setup Instructions (Linux)

1. Clone the repository:
```bash
git clone https://github.com/prabhat51/image-segmentation-assignment.git
cd image-segmentation-assignment
```

2. Install dependencies using `uv`:
```bash
pip install uv
uv pip install -r requirements.txt
```

---

## 🧪 Task 1: Dataset Preparation

Run the preprocessing script:

```bash
python data_preprocessing.py \
  --annotations path/to/annotations.json \
  --images path/to/images \
  --output ./dataset \
  --max-images 5000
```

### ✅ Features
- Converts COCO JSON annotations into multi-class masks
- Handles missing annotations, I/O errors, invalid masks
- Stores output in `dataset/images/` and `dataset/masks/`

---

## 🧠 Task 2: Model Training

Run the training script:

```bash
python train.py \
  --data_path ./dataset \
  --epochs 50 \
  --lr 1e-4 \
  --num_classes 80 \
  --batch_size 16 \
  --weight_decay 0.01 \
  --img_size 512,512
```

### ⚙️ Configuration Options
- Image resolution, learning rate, number of classes
- Supports training with mixed precision on GPU
- Logs metrics with **Weights & Biases** or TensorBoard

---

## 📊 Evaluation & Inference

Metrics used:
- Intersection over Union (IoU)
- Dice Coefficient
- Pixel Accuracy

Model checkpoints are saved in `checkpoints/`. Inference scripts can reuse trained models for prediction and visualization.


---

## ✅ Submission Guidelines

- GitHub repo with code, README, and sample visualizations
- Example masks and predictions must be included
- Report summarizing:
  - Design choices
  - Edge case handling
  - Model architecture and improvements
  - Training resources and time
- Reproducibility from command line on Linux using `uv`


> **Note:** Do not use Ultralytics (YOLOv8) for this assignment. Focus is on approach, reproducibility, and code clarity — not on final accuracy.

---

