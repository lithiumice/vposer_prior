
# Human Body Prior

This repository is a modified version of [VPoser: Variational Human Pose Prior for Body Inverse Kinematics](https://github.com/nghorbani/human_body_prior), adapted to work as a pip library with additional features and improvements.

## Roadmap

- [ ] Add compatibility with VPoser v1.0
- [ ] Integrate AMASS repository support

## Installation Guide

### Prerequisites

#### 1. SMPL-X Model Setup
```bash
# Download SMPL-X neutral model
mkdir -p data/smplx/
wget "https://huggingface.co/lithiumice/models_hub/resolve/main/smpl_smplh_smplx_mano/SMPLX_NEUTRAL.npz" -O data/smplx/SMPLX_NEUTRAL.npz

# Download SMPL neutral model
mkdir -p data/smpl/
wget "https://huggingface.co/lithiumice/models_hub/resolve/main/smpl_smplh_smplx_mano/SMPL_NEUTRAL.pkl" -O data/smpl/SMPL_NEUTRAL.pkl
```

#### 2. Environment Setup

Create conda environment:
```bash
conda env create -f _base_conda_env.yaml -n torch_base
```

Install PyTorch3D:
```bash
conda install pytorch3d -c pytorch3d
```

### Installation Methods

Choose one of the following installation methods:

#### Option 1: Development Installation
```bash
git clone https://github.com/lithiumice/vposer_prior
cd vposer_prior 
pip install -e .
```

#### Option 2: Poetry Installation (Recommended)
```bash
pip install poetry
poetry install
```

#### Option 3: Direct Installation
From GitHub using poetry:
```bash
poetry add "git+https://github.com/lithiumice/vposer_prior"
```

Or using pip:
```bash
pip install "git+https://github.com/lithiumice/vposer_prior"
pip install "git+https://github.com/mattloper/chumpy"
```

## Usage Guide

### VPoser Body Prior

1. Download VPoser model weights:
```bash
# Note: This repository only supports VPoser v2
git clone https://huggingface.co/lithiumice/vposer_v02_05 data/vposer_v02_05
```

2. Python implementation:
```python
from human_body_prior.tools.model_loader import exprdir2model

vposer, _ = exprdir2model("data/vposer_v02_05")
```

### 3D SMPL Visualization

#### Setup Requirements

1. Install visualization dependencies:
```bash
# Choose appropriate PyVista version:
pip install pyvista==0.44.1  # Latest compatible version
# OR
pip install pyvista==0.35.2  # Legacy version

# Additional requirements
pip install pyrender pyvista seaborn
sudo apt install libgl1-mesa-glx xvfb
```

2. Test visualization setup:
```python
import pyvista as pv

mesh = pv.Sphere()
p = pv.Plotter()
p.add_mesh(mesh)
mesh.plot()
p.show()
```

3. Test SMPL visualizer:
```bash
export PYOPENGL_PLATFORM=osmesa
python tests/test_smpl_3d_vis.py
```

## Troubleshooting

### Known Issues

**Issue**: ImportError: cannot import name 'OSMesaCreateContextAttribs' from 'OpenGL.osmesa'

**Solution**:
```bash
pip install --upgrade pyopengl==3.1.4
```        
