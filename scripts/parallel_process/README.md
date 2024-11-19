
# Render SMPL motion npz batch with ray

    pip install ray hydra-core

    ln -s ~/2_data/0_SmplBodyModels/smpl_npz_SMPL_N_model_npz ./data/smpl
    ln -s ~/2_data/0_SmplBodyModels/smpl_pkl_SMPL_NEUTRAL_pkl ./data/smpl

    python scripts/parallel_process/ray_main.py --config scripts/parallel_process/configs/local_rtx3060/20240821_loco_vis.yaml

    python scripts/parallel_process/ray_main.py --config scripts/parallel_process/configs/local_rtx3060/WSL_dance.yaml
