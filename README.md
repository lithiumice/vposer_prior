# README
The original repo is [VPoser: Variational Human Pose Prior for Body Inverse Kinematics](https://github.com/nghorbani/human_body_prior), and this is for some modifications and adoptions of pip library.

## TODO
- [] Add compatible to Vposer v1.0
- [] Add AMASS repo

## Usage

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

## Install
<!-- ```bash
conda activate difftraj
pip install human-body-prior
``` -->

Install this version:
1. install without cloning
```bash
pip install "git+https://github.com/lithiumice/human_body_prior"
```

2. clone and install
```bash
# pip uninstall human_body_prior -y

git clone https://github.com/lithiumice/human_body_prior

cd human_body_prior 
python setup.py install

pip show human_body_prior
```

3. install with poetry[Recommand]
```bash
pip install poetry
poetry install
pip install -e .

# or
poetry add "git+https://github.com/lithiumice/human_body_prior"
```
This will install in editable way.

Check you had install these pip package, if you do not mess up your environment:
+ torch
+ pytorch3d

LICENSE: license from MPI, no free to modify and distribut,