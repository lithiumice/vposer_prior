
import ray
import sys
from glob import glob
import argparse

from omegaconf import OmegaConf
from pathlib import Path
from ray.util.actor_pool import ActorPool

import os
import sys
import torch

from body_visualizer.vis3d.visualizer3d_smpl import SMPLVisualizer
import numpy as np
import torch


if __name__ == "__main__":
    output_folder = "outputs/"
    npz_path = "data/demo_npzs/taijiquan_female_ID0_difTraj_raw.npz"
    render_video_path = str(Path(output_folder, Path(npz_path).stem + ".mp4"))
    

    visualizer = SMPLVisualizer(generator_func=None, distance=7, device="cuda", enable_shadow=True, anti_aliasing=True,
                                smpl_model_dir="data/smpl", sample_visible_alltime=True, verbose=True, enable_cam_following=False)
    
    
    print(f"Processing {npz_path}")
    data = np.load(npz_path)
    
    # shape: num_persion, T, ...
    common_op = lambda x: x[:].unsqueeze(0)

    smpl_seq = {
        'pose': common_op(torch.tensor(data['poses'])[:,:24,:].reshape(-1,72)),
        "trans": common_op(torch.tensor(data["trans"])),
        "shape": common_op(torch.tensor(data["betas"])),
    }
    
    init_args = {'smpl_seq': smpl_seq, 'mode': 'gt'}
    visualizer.save_animation_as_video(render_video_path, init_args=init_args, window_size=(1500, 1500), cleanup=True)
    print(f"[INFO] Save video to {render_video_path}")
    