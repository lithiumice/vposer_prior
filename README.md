
# Human Body Prior

[![PyPI version](https://badge.fury.io/py/human_body_prior_v2.svg)](https://badge.fury.io/py/human_body_prior_v2)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive Python library for human body pose modeling and visualization, based on VPoser (Variational Human Pose Prior). This library provides tools for human pose synthesis, estimation, and 3D body visualization with SMPL/SMPL-X body models.

## ✨ Features

- **VPoser Integration**: State-of-the-art variational human pose prior for body inverse kinematics
- **3D Visualization**: Integrated tools for SMPL/SMPL-X body model visualization
- **Multiple Installation Methods**: Support for pip, poetry, and development installations
- **AMASS Dataset Support**: Compatible with AMASS motion capture dataset
- **GPU Acceleration**: Optimized for CUDA with PyTorch backend
- **Easy to Use**: Simple API for quick integration into research projects

## 🚀 Quick Start

```bash
# Install with poetry (recommended)
pip install poetry
poetry add "git+https://github.com/lithiumice/vposer_prior"

# Or install with pip
pip install "git+https://github.com/lithiumice/vposer_prior"
```

```python
from human_body_prior.model_hub import get_vposer_model

# Load VPoser model
vposer = get_vposer_model(device='cuda', vposer_ckpt='data/vposer_v02_05')

# Encode body poses
latent_codes = vposer.encode(body_poses)
reconstructed_poses = vposer.decode(latent_codes)
```

## 📋 Roadmap

- [ ] Add compatibility with VPoser v1.0
- [ ] Integrate AMASS repository support
- [ ] Add PyTorch Lightning integration
- [ ] Improve documentation with more examples

## 📦 Installation

### System Requirements

- Python 3.7+
- CUDA-enabled GPU (recommended for optimal performance)
- 8GB+ RAM
- Linux/macOS (Windows support via WSL)

### Step 1: Download Body Models

```bash
# Create data directories
mkdir -p data/smplx/ data/smpl/

# Download SMPL-X neutral model
wget "https://huggingface.co/lithiumice/models_hub/resolve/main/smpl_smplh_smplx_mano/SMPLX_NEUTRAL.npz" -O data/smplx/SMPLX_NEUTRAL.npz

# Download SMPL neutral model  
wget "https://huggingface.co/lithiumice/models_hub/resolve/main/smpl_smplh_smplx_mano/SMPL_NEUTRAL.pkl" -O data/smpl/SMPL_NEUTRAL.pkl
```

### Step 2: Environment Setup

#### Option A: Conda Environment (Recommended)

```bash
# Create base environment
conda env create -f scripts/installation/_base_conda_env.yaml -n torch_base
conda activate torch_base

# Install PyTorch3D
conda install pytorch3d -c pytorch3d
```

#### Option B: Manual Setup

```bash
# Create conda environment
conda create -n vposer python=3.8
conda activate vposer

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pyyaml tqdm dotmap transforms3d omegaconf loguru
```

### Step 3: Install VPoser Prior

Choose your preferred installation method:

#### 🎯 Poetry (Recommended)
```bash
# Install poetry if not already installed
pip install poetry

# Clone and install
git clone https://github.com/lithiumice/vposer_prior
cd vposer_prior
poetry install
```

#### 🔧 Development Mode
```bash
git clone https://github.com/lithiumice/vposer_prior
cd vposer_prior
pip install -e .
```

#### 📦 Direct from GitHub
```bash
# Using Poetry
poetry add "git+https://github.com/lithiumice/vposer_prior"

# Using pip
pip install "git+https://github.com/lithiumice/vposer_prior"
pip install "git+https://github.com/mattloper/chumpy"
```

## 💻 Usage Guide

### Download VPoser Models

First, download the VPoser model weights:

```bash
# Download VPoser v2.0.5 model (recommended)
git clone https://huggingface.co/lithiumice/vposer_v02_05 data/vposer_v02_05

# Alternative SSH URL
git clone git@hf.co:lithiumice/vposer_v02_05 data/vposer_v02_05
```

### 🎯 Basic VPoser Usage

#### Load VPoser Model

```python
from human_body_prior.model_hub import get_vposer_model

# Load VPoser model on GPU
vposer = get_vposer_model(device='cuda', vposer_ckpt='data/vposer_v02_05')
print(f"VPoser model loaded: {vposer}")
```

#### Encode and Decode Body Poses

```python
import torch
import numpy as np

# Load sample pose data (from AMASS dataset)
sample_amass_fname = "data/support_data/dowloads/amass_sample.npz"
amass_body_pose = np.load(sample_amass_fname)['poses'][:, 3:66]  # Body joints only
amass_body_pose = torch.from_numpy(amass_body_pose).float().to('cuda')

print(f"Body pose shape: {amass_body_pose.shape}")  # (N, 63)

# Encode poses to latent space
latent_result = vposer.encode(amass_body_pose)
latent_codes = latent_result.mean  # or use latent_result.sample()
print(f"Latent codes shape: {latent_codes.shape}")

# Decode latent codes back to poses
reconstructed_poses = vposer.decode(latent_codes)
print(f"Reconstructed poses shape: {reconstructed_poses.shape}")
```

#### Generate Novel Poses

```python
# Sample random latent codes
num_samples = 10
latent_dim = 32  # VPoser latent dimension
random_latent = torch.randn(num_samples, latent_dim).to('cuda')

# Generate poses from latent codes
generated_poses = vposer.decode(random_latent)
print(f"Generated poses shape: {generated_poses.shape}")
```

### 🎨 3D Visualization

#### Install Visualization Dependencies

```bash
# Install PyVista (choose version based on your system)
pip install pyvista==0.44.1  # Latest stable version
# OR
pip install pyvista==0.35.2  # Legacy version for older systems

# Additional visualization packages
pip install pyrender seaborn

# System dependencies for headless rendering
sudo apt-get install libgl1-mesa-glx xvfb
```

#### Test Visualization Setup

```python
import pyvista as pv

# Test basic PyVista functionality
mesh = pv.Sphere()
plotter = pv.Plotter(off_screen=True)  # Use off_screen for headless
plotter.add_mesh(mesh)
plotter.show(screenshot='test_sphere.png')
print("PyVista setup successful!")
```

#### Visualize SMPL Body Models

```python
from body_visualizer.tools.vis_tools import show_smpl_body
import torch

# Generate some body poses
body_poses = torch.randn(1, 63).to('cuda')  # 21 joints * 3 = 63
body_shape = torch.randn(1, 10).to('cuda')   # SMPL shape parameters

# Visualize (set environment variable first)
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

show_smpl_body(body_poses, body_shape, save_path='body_visualization.png')
```

### 🔧 Advanced Usage

#### Inverse Kinematics with VPoser

```python
from human_body_prior.models.ik_engine import IKEngine

# Create IK engine
ik_engine = IKEngine(vposer_model=vposer)

# Define target joint positions
target_joints = torch.randn(1, 21, 3).to('cuda')  # 21 joints in 3D

# Solve for body pose
result_pose = ik_engine.solve(target_joints)
print(f"IK result shape: {result_pose.shape}")
```

#### AMASS Dataset Integration

```python
from human_body_prior.data.prepare_data import load_amass_data

# Load AMASS dataset
amass_data = load_amass_data('path/to/amass/dataset')

# Process motion sequences
for sequence in amass_data:
    poses = sequence['poses'][:, 3:66]  # Extract body poses
    latent_codes = vposer.encode(poses)
    # Further processing...
```

### 🧪 Testing Your Installation

```bash
# Test VPoser functionality
python -c "from human_body_prior.model_hub import get_vposer_model; print('VPoser import successful')"

# Test visualization (requires GUI or headless setup)
export PYOPENGL_PLATFORM=osmesa
python scripts/tutorials/demo.py

# Run unit tests
python scripts/unit_tests/test_vposer.py
python scripts/unit_tests/test_rotations.py
```

## 📁 Project Structure

```
vposer_prior/
├── src/
│   ├── human_body_prior_v2/          # Core VPoser library
│   │   ├── body_model/              # SMPL/SMPL-X body models
│   │   ├── models/                  # VPoser model implementations
│   │   ├── tools/                   # Utilities and helpers
│   │   ├── train/                   # Training scripts
│   │   └── visualizations/          # Visualization tools
│   └── body_visualizer/             # 3D body visualization
│       ├── mesh/                    # Mesh processing
│       ├── tools/                   # Visualization utilities
│       └── vis3d/                   # 3D rendering
├── data/                            # Data and model files
│   ├── smpl/                       # SMPL body models
│   ├── smplx/                      # SMPL-X body models
│   ├── vposer_v02_05/              # VPoser v2.0.5 model
│   └── support_data/               # Sample data and assets
├── scripts/
│   ├── tutorials/                  # Tutorial notebooks and examples
│   ├── unit_tests/                 # Unit tests
│   ├── data_analysis/              # Data analysis tools
│   └── parallel_process/           # Parallel processing utilities
└── docs/                           # Documentation
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/lithiumice/vposer_prior
cd vposer_prior

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest scripts/unit_tests/

# Format code
black src/
flake8 src/
```

### Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Use type hints where appropriate

## 📚 Documentation

- **API Reference**: [Coming Soon]
- **Tutorials**: Check the `scripts/tutorials/` directory
- **Examples**: See `scripts/tutorials/` for practical examples
- **VPoser Paper**: [VPoser: Variational Human Pose Prior](https://arxiv.org/abs/1904.05866)

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size or use CPU
vposer = get_vposer_model(device='cpu')
```

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r scripts/installation/requirements/base.txt
pip install "git+https://github.com/mattloper/chumpy"
```

#### Visualization Issues
```bash
# Set environment variable for headless rendering
export PYOPENGL_PLATFORM=osmesa

# Install system dependencies
sudo apt-get install libgl1-mesa-glx xvfb
```

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/lithiumice/vposer_prior/issues)
- **Documentation**: Check the `scripts/tutorials/` directory for examples
- **Original VPoser**: [Reference implementation](https://github.com/nghorbani/human_body_prior)

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Original VPoser**: [Nima Ghorbani](https://nghorbani.github.io/) and the Max Planck Institute team
- **SMPL Body Models**: [SMPL/SMPL-X](https://smpl.is.tue.mpg.de/) by the Max Planck Institute
- **AMASS Dataset**: [AMASS](https://amass.is.tue.mpg.de/) for motion capture data

## 📞 Citation

If you use this library in your research, please cite the original VPoser paper:

```bibtex
@inproceedings{ghorbani2019vposer,
  title={VPoser: Variational Human Pose Prior for Body Inverse Kinematics},
  author={Ghorbani, Nima and Black, Michael J.},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
