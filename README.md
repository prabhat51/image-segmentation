🧠 Image Segmentation Assignment
This project implements an end-to-end pipeline for image segmentation using PyTorch. It covers dataset preparation, training a segmentation model, and visualizing results. The assignment was designed to evaluate my efficiency in data preprocessing, model training, and making informed design choices under resource constraints.

📌 Overview
This repository contains two main components:

Task 1 – Dataset preparation: Scripts to process a dataset (e.g., COCO or Cityscapes) into segmentation masks.

Task 2 – Model training: A PyTorch-based implementation of an image segmentation model (e.g., UNet or DeepLabV3), trained and evaluated with metrics.

🧰 Setup Instructions (Linux)
Clone the repository:
bash
Copy
Edit
git clone https://github.com/<your-username>/image-segmentation-assignment.git
cd image-segmentation-assignment
Install dependencies using uv:
bash
Copy
Edit
pip install uv
uv pip install -r requirements.txt
🧪 Task 1: Dataset Preparation
Features:
Accepts raw images and annotations (JSON/XML).

Generates binary/multi-class masks.

Handles overlapping regions and invalid annotations.

Works with 3,000–8,000 images.

Edge Case Handling (Documented Here):
Overlapping regions are resolved using instance priority.

Missing annotations are ignored with warnings.

Classes outside predefined labels are discarded.

Images without matching annotations are skipped.

Empty masks are removed from training set.

Files with I/O errors are logged and skipped.

Cropping/resizing ensures consistent shape.

Invalid polygons are auto-corrected.

Unsupported image formats are filtered.

Dataset integrity is checked before saving.

🧠 Task 2: Model Training
Model:
Architecture: [e.g., UNet / DeepLabV3 / Vision Transformer]

Trained using PyTorch.

Designed for generalization across unseen data.

Trained under 6 hours compute time.

Evaluation Metrics:
IoU (Intersection over Union)

Dice Coefficient

Pixel Accuracy

Monitoring:
Training progress is logged using WandB or TensorBoard.

📊 Results & Inference
Visual samples of predicted masks are available in the results/ directory.

Inference script: inference.py (loads model weights and outputs masks).
