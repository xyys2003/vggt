VGGT → COLMAP → Gaussian Splatting (3DGS) Example

This repository is a minimal example demonstrating how to:

Run VGGT to estimate camera poses and depth.

Export the predictions into COLMAP format.

Use the exported results for Gaussian Splatting / 3DGS pipelines such as gsplat.

Only the script demo_colmap.py is retained.
All unrelated files and documentation from the original VGGT repo have been removed to keep this repository lightweight and focused.

📦 Installation

Create a clean environment and install VGGT + required dependencies:

# Create environment
conda create -n vggt python=3.10 -y
conda activate vggt

# Install PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install dependencies for demo_colmap.py
pip install -r requirements.txt

📤 Exporting VGGT Predictions to COLMAP

demo_colmap.py converts VGGT outputs into standard COLMAP sparse format:

SCENE_DIR/
├── images/          # input images
└── sparse/          # COLMAP results
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin

▶️ Run VGGT (feed-forward only)
python demo_colmap.py \
    --scene_dir /PATH/TO/SCENE_DIR

▶️ With Bundle Adjustment (recommended)
python demo_colmap.py \
    --scene_dir /PATH/TO/SCENE_DIR \
    --use_ba

▶️ Faster BA (lower matching cost)
python demo_colmap.py \
    --scene_dir /PATH/TO/SCENE_DIR \
    --use_ba \
    --max_query_pts 2048 \
    --query_frame_num 5


After running any of these commands, COLMAP files will appear in:

/PATH/TO/SCENE_DIR/sparse/


These files can be directly consumed by gsplat.

🌀 Reconstruct COLMAP Data with Gaussian Splatting (gsplat)

Once demo_colmap.py has finished, your scene directory should look like:

SCENE_DIR/
├── images/          # input images
└── sparse/          # COLMAP outputs from VGGT
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin

1️⃣ Install gsplat

We recommend using the stable version:

pip install "gsplat==1.3.0"


or install from source:

git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
pip install -e .

2️⃣ Train a Gaussian Splatting model

Simply point gsplat to your VGGT-generated scene:

cd gsplat

python examples/simple_trainer.py default \
    --data_factor 1 \
    --data_dir /PATH/TO/SCENE_DIR \
    --result_dir /PATH/TO/RESULT_DIR

Arguments
Argument	Meaning
--data_dir	Directory containing images/ and sparse/
--result_dir	Output directory for results, checkpoints, and renders
--data_factor	Downsampling factor for input images (1 = full res)
🔄 Full Pipeline (End-to-End)
# Step 1: VGGT → COLMAP
python demo_colmap.py --scene_dir /SCENE --use_ba

# Step 2: Install gsplat
pip install "gsplat==1.3.0"

# Step 3: Gaussian Splatting
cd gsplat
python examples/simple_trainer.py default \
    --data_factor 1 \
    --data_dir /SCENE \
    --result_dir /SCENE/output