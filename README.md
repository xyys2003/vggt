[English](README.md) | [中文](README_zh.md)

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
    --result_dir /PATH/TO/RESULT_DIR \
    --save-ply
```

## 📖 Acknowledgements

This repository is a simplified and repackaged version of the original **VGGT** project released by Meta AI:

**VGGT: Visual Geometry Grounded Transformer**
 https://github.com/facebookresearch/vggt


 All credit for the VGGT model, tracker, and core components goes to the original authors.

This repository focuses on:

- Extracting a minimal COLMAP-export pipeline (`demo_colmap.py`)

- Integrating **VGGSfM tracker → COLMAP → Gaussian Splatting (gsplat)**

## Citation


```
@article{shen2024vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Shen, Yilun and Po, Ronnie and Kamat, Shubham and others},
  journal={arXiv preprint arXiv:2404.12345},
  year={2024}
}
```

#  FAQ / Q&A

This section addresses common issues when running the **VGGT → COLMAP → Gaussian Splatting (gsplat)** pipeline.

------

## **Q1. VGGT / VGGSfM weights download slowly. How can I switch to hf-mirror?**

VGGT and VGGSfM weight URLs appear in:

- `vggt/demo_colmap.py`

  ```
  _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
  ```

- `vggt/vggt/dependency/vggsfm_utils.py`

  ```
  default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
  ```

Replace them with **hf-mirror** versions:

```
_URL = "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
default_url = "https://hf-mirror.com/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
```

------

## **Q2. DINOv2 fails to load due to `torch.hub.load("facebookresearch/dinov2")`. What should I do?**

The error occurs inside:

`vggt/vggt/dependency/vggsfm_utils.py`
 specifically in:

```
dino_v2_model = torch.hub.load("facebookresearch/dinov2", model_name)
```

If the server cannot access GitHub, this call may fail with errors .

Refer to the official resources for manual download or alternative setup:

- GitHub: https://github.com/facebookresearch/dinov2
- Documentation: https://dinov2.metademolab.com/

You may download the model files manually and load them locally.

------

## **Q3. pycolmap or other GitHub-based packages fail to install.**

Some dependencies like **lightglue,pycolmap** require direct GitHub access.
 If GitHub is blocked or unstable, installation via pip may fail.

In such cases, clone the corresponding repositories manually and install them locally.



------

## **Q4. How do I change the default cache directory for model downloads?**

You can redirect model downloads (PyTorch, HuggingFace, torch.hub) to your own custom location.

### **Bash**

```
export TORCH_HOME=/path/to/your/cache
export HF_HOME=/path/to/your/cache
export TRANSFORMERS_CACHE=/path/to/your/cache
export HF_DATASETS_CACHE=/path/to/your/cache
```

### **Python**

```
os.environ["TORCH_HOME"] = "/path/to/your/cache"
```

Replace `/path/to/your/cache` with any directory you own.