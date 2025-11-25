[English](README.md) | [中文](README_zh.md)

# README_zh.md

## VGGT → COLMAP → Gaussian Splatting示例

本仓库提供一个最小可运行的端到端示例，展示以下完整流程：

1. 使用 VGGT 预测相机位姿与深度
2. 将预测结果导出至 COLMAP 稀疏格式
3. 使用 gsplat 训练 Gaussian Splatting 模型

仓库已经包含 gsplat 作为 git submodule，因此 clone 完成本仓库后即可运行整个 3DGS 训练流程。

------

## 安装步骤

### 1. 克隆仓库（必须包含子模块）

```
git clone --recursive https://github.com/xyys2003/vggt.git
cd vggt
```

### 2. 创建环境并安装 VGGT 依赖

```
conda create -n vggt python=3.10 -y
conda activate vggt
```

安装 PyTorch（例如 CUDA 12.4）：

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

安装 VGGT 所需依赖：

```
pip install -r requirements.txt
```

------

## 导出 VGGT 结果至 COLMAP

运行 `demo_colmap.py` 后，将自动写出 COLMAP 稀疏格式结构：

```
SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### 基本重建

```
python demo_colmap.py --scene_dir /PATH/TO/SCENE_DIR
```

### use_ba

```
python demo_colmap.py --scene_dir /PATH/TO/SCENE_DIR --use_ba
```

### 显存不够可降低参数

```
python demo_colmap.py \
    --scene_dir /PATH/TO/SCENE_DIR \
    --use_ba \
    --max_query_pts 2048 \
    --query_frame_num 5
```

------

## Gaussian Splatting（gsplat）

确保你已经拥有标准场景结构：

```
SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### 1. 安装 gsplat

VGGT 与 gsplat 使用的 pycolmap 版本不同，因此推荐使用独立环境：

```
conda create -n gsplat python=3.10 -y
conda activate gsplat
```

进入 gsplat 子模块：

```
cd gsplat
```

安装 PyTorch：

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

构建 gsplat：

```
pip install -e . --no-build-isolation
```

安装示例训练依赖：

```
pip install -r examples/requirements.txt --no-build-isolation
```

------

### 2. 训练 Gaussian Splatting 模型

```
python examples/simple_trainer.py default \
    --data_factor 1 \
    --data_dir /PATH/TO/SCENE_DIR \
    --result_dir /PATH/TO/RESULT_DIR \
    --save-ply
```

------

## 致谢

本仓库源自 Meta AI 发布的原始 VGGT 项目：

**VGGT: Visual Geometry Grounded Transformer**
 https://github.com/facebookresearch/vggt

本仓库主要聚焦于：

- 提供最小可运行的 COLMAP 导出pipeline（demo_colmap.py）
- 集成 VGGSfM → COLMAP → Gaussian Splatting（gsplat）

------

## Citation

```
@article{shen2024vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Shen, Yilun and Po, Ronnie and Kamat, Shubham and others},
  journal={arXiv preprint arXiv:2404.12345},
  year={2024}
}
```