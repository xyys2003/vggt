# VGGT → COLMAP → Gaussian Splatting (3DGS) Example

This repository provides a **minimal end-to-end example** demonstrating:

1. Running **VGGT** to estimate camera poses & depth  
2. Exporting outputs into **COLMAP sparse format**  
3. Training a **Gaussian Splatting model** using **gsplat**

The repository already includes **gsplat as a git submodule**, so the full 3DGS training pipeline can be executed after a single clone.

---

## 📦 Installation

### **1️⃣ Clone this repository (with submodules)**  
⚠️ Mandatory — gsplat depends on its own submodules (e.g., glm):

```bash
git clone --recursive https://github.com/xyys2003/vggt.git
cd vggt
```

2️⃣ Create environment & install VGGT dependencies

```bash
conda create -n vggt python=3.10 -y
conda activate vggt


##Install PyTorch (For Example : CUDA 12.4):

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124


##Install VGGT dependencies:

pip install -r requirements.txt
```

## 📤 Exporting VGGT Predictions to COLMAP
```bash
# demo_colmap.py writes VGGT outputs into standard COLMAP structure:

SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### ▶️ Feed-forward reconstruction
```bash
python demo_colmap.py --scene_dir /PATH/TO/SCENE_DIR
```
```bash
### ▶️ With Bundle Adjustment (recommended)
python demo_colmap.py --scene_dir /PATH/TO/SCENE_DIR --use_ba
```

```bash
### ▶️ Faster BA (lower matching cost)
python demo_colmap.py \
    --scene_dir /PATH/TO/SCENE_DIR \
    --use_ba \
    --max_query_pts 2048 \
    --query_frame_num 5
```





## 🌀 Gaussian Splatting (gsplat)
```bash
# After exporting COLMAP, your scene directory should contain:

SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### 1️⃣ Install gsplat 
```bash
#VGGT (demo_colmap) and gsplat require different pycolmap versions.
#Mixing them in the same environment will cause import errors.
conda create -n gsplat python=3.10 -y

conda activate gsplat

cd gsplat

# Install PyTorch first
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Build gsplat with local CUDA/PyTorch
pip install -e . --no-build-isolation

# Install training/example dependencies
pip install -r examples/requirements.txt --no-build-isolation

```


### 2️⃣ Train a Gaussian Splatting model
```bash
python examples/simple_trainer.py default \
    --data_factor 1 \
    --data_dir /PATH/TO/SCENE_DIR \
    --result_dir /PATH/TO/RESULT_DIR
```