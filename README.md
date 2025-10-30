# design-project

# PASCAL VOC 2012 Split 3 Semantic Segmentation

This project performs semantic segmentation on PASCAL VOC 2012 dataset (Split 3) using PSPNet with ResNet backbone.

---

## Model Architecture (Cells 1-4)

**File:** `model_architecture.py`

**What's included:**
- Installation commands
- All imports
- ResNet building blocks (conv3x3, Bottleneck, ResNet)
- PSPNet architecture (PPM, PSPNet)
- Dataset configuration (Split 3 classes, color map)

**Lines:** ~330 lines

---

## Model Loading & Preprocessing (Cells 5-6)

**File:** `model_loading_preprocessing.py`

**What's included:**
- Model loading function
- Checkpoint handling
- File upload functionality
- Image preprocessing functions
- Normalization and resizing logic

**Lines:** ~140 lines

---

## Inference & Visualization (Cells 7-11)

**File:** `inference_visualization.py`

**What's included:**
- Segmentation inference function
- Visualization functions (with labels, overlays)
- Statistics printing
- Batch processing
- Result saving and download

**Lines:** ~200 lines

---

## Simple Project Structure

```
pascal-voc-segmentation/
├── README.md
├── model_architecture.py
├── model_loading_preprocessing.py
└── inference_visualization.py
 
```

---

## How to Use


```python
# main_inference.py
from model_architecture import *
from model_loading_preprocessing import *
from inference_visualization import *

# Now run the complete pipeline
# Upload model -> Load model -> Upload image -> Run inference
```

Or in Google Colab, run three cells:
```python
# Cell 1: 
%run model_architecture.py

# Cell 2: 
%run model_loading_preprocessing.py

# Cell 3:
%run inference_visualization.py
```

---

## Requirements.txt

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python-headless>=4.5.0
pillow>=8.0.0
matplotlib>=3.3.0
numpy>=1.19.0
```

---


## Model Configuration

- **Classes:** 16 (15 base + background)
- **Architecture:** PSPNet + ResNet50
- **Input Size:** 473x473
- **Split 3 Base Classes:** airplane, bird, bottle, bus, cat, chair, cow, dining table, horse, person, potted plant, sheep, sofa, train, tv

## Contributors

- Model Architecture
- Data Processing
- Inference Pipeline

---

