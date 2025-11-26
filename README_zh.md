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

# Q&A 常见问题解答

## Q1. VGGT / VGGSfM 权重下载缓慢，如何切换到 hf-mirror？

VGGT 和 VGGSfM 的权重下载链接分别位于：

- `vggt/demo_colmap.py`

```
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
```

- `vggt/vggt/dependency/vggsfm_utils.py`

```
default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
```

如果服务器访问 huggingface 较慢，可以将其替换为 hf-mirror 镜像：

```
_URL = "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
default_url = "https://hf-mirror.com/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
```

替换后，模型会从镜像源下载，速度更快且稳定。

------

## Q2. DINOv2 加载失败，提示 torch.hub 无法访问 GitHub，怎么办？

在文件 `vggt/vggt/dependency/vggsfm_utils.py` 中包含以下代码：

```
dino_v2_model = torch.hub.load("facebookresearch/dinov2", model_name)
```

如果服务器无法访问 GitHub，则 torch.hub 会报错。

可选解决方法：

1. 手动下载 DINOv2 仓库与权重，并在本地加载
2. 配置代理或镜像以确保能访问 GitHub
3. 从官方仓库获取：

- GitHub: https://github.com/facebookresearch/dinov2
- Documentation: https://dinov2.metademolab.com/

------

## Q3. 安装 pycolmap 或其他 GitHub 来源的依赖时报错怎么办？

一些依赖（例如 LightGlue、pycolmap、部分结构化匹配工具）需要直接从 GitHub 下载源码。

如果服务器无法访问 GitHub，可以采用以下方案：

1. 手动克隆相关仓库并本地安装
2. 在本地已有网络环境中下载源码，再上传到服务器
3. 使用镜像服务（如 ghproxy、ghfast、kgithub）

------

## Q4. 如何修改默认的模型缓存目录？

如果你不希望模型下载到默认的 `~/.cache` 目录，可以通过环境变量修改 PyTorch、HuggingFace 等组件的缓存路径。

### Bash 方式：

```
export TORCH_HOME=/path/to/your/cache
export HF_HOME=/path/to/your/cache
export TRANSFORMERS_CACHE=/path/to/your/cache
export HF_DATASETS_CACHE=/path/to/your/cache
```

### Python 方式：

```
os.environ["TORCH_HOME"] = "/path/to/your/cache"
```

其中 `/path/to/your/cache` 是你希望放置模型文件的路径。