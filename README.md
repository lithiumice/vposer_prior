# README
The original repo is [VPoser: Variational Human Pose Prior for Body Inverse Kinematics](https://github.com/nghorbani/human_body_prior), and this is for some modifications and adoptions of pip library.

## TODO
- [] Add compatible to Vposer v1.0
- [] Add AMASS repo

## Environment

    conda env create -f _base_conda_env.yaml -n torch_base

## Usage

### Vposer

download vposer model weight

```bash
# this repo can only use vposer v2
# git clone https://huggingface.co/lithiumice/vposer_v1_0 data/vposer_v1_0

git clone https://huggingface.co/lithiumice/vposer_v02_05 data/vposer_v02_05
```

used in python

```python
from human_body_prior.tools.model_loader import exprdir2model

vposer, _ = exprdir2model("data/vposer_v02_05")
```

### 3D SMPL visualizer

Install Requirements

    # old version
    pip install pyvista==0.35.2

    # new version to compatible for pyvista changes
    pip install pyvista==0.44.1

    pip install pyrender pyvista

test pyvista:

    import pyvista as pv

    pv.start_xvfb()
    mesh = pv.Sphere()
    p = pv.Plotter()
    p.add_mesh(mesh)
    mesh.plot()
    
    p.show()

Test SMPL visualizer

    python debug_scripts/test_vis_3d.py


### Download SMPL-X model

    mkdir data/smplx/
    wget "https://huggingface.co/lithiumice/SMPLit/resolve/main/smpl_smplh_smplx_mano/SMPLX_NEUTRAL.npz" -O data/smplx/SMPLX_NEUTRAL.npz

    mkdir data/smpl/
    wget "https://huggingface.co/lithiumice/SMPLit/resolve/main/smpl_smplh_smplx_mano/SMPL_NEUTRAL.pkl" -O data/smpl/SMPL_NEUTRAL.pkl

## Install

clone and install

    git clone https://github.com/lithiumice/human_body_prior
    cd human_body_prior 
    pip install -e .
    pip show human_body_prior

install with poetry[Recommand]

    pip install poetry
    poetry install

install from github using poetry

    poetry add "git+https://github.com/lithiumice/human_body_prior"
    
install from github using pip
    
    pip install "git+https://github.com/lithiumice/human_body_prior"
    pip install "git+https://github.com/mattloper/chumpy"


This will install in editable way.

Check you had install these pip package, if you do not mess up your environment:
+ torch
+ pytorch3d

LICENSE: license from MPI, no free to modify and distribut,

## Issues

- ImportError: cannot import name 'OSMesaCreateContextAttribs' from 'OpenGL.osmesa
    - pip install --upgrade pyopengl==3.1.4