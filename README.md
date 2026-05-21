# Vertebral Fracture Detection Training

## Directory Structure

```
/
├── train20251001.py
├── best.pt              # YOLO model weights for vertebra detection
└── data/
    ├── fracture/
    │   ├── patient_001/
    │   │   ├── front.jpg
    │   │   └── side.jpg
    │   └── ...
    └── non_fracture/
        ├── patient_001/
        │   ├── front.jpg
        │   └── side.jpg
        └── ...
```

## Requirements

```bash
pip install numpy opencv-python matplotlib scikit-learn tensorflow ultralytics
```

## Setup

1. Create `data/fracture/` and `data/non_fracture/` directories.
2. Each patient subdirectory should contain two JPEG images (frontal and lateral radiographs), sorted alphabetically (frontal first).

## Usage

```bash
python train20251001.py
```

Training logs and model weights are saved to `log/<timestamp>/`.
