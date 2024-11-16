from body_visualizer.vis3d.pyrender_smpl import SMPLVisualizer


import os
import sys
import torch

import numpy as np
import torch




visualizer = SMPLVisualizer(
    generator_func=None, device="cuda", show_smpl=True, 
    smpl_model_dir="data/smpl", show_skeleton=False, sample_visible_alltime=False, 
    
    init_T=6, enable_shadow=False, anti_aliasing=True, use_floor=True,
    distance=7, elevation=20, azimuth=30, verbose=True
)


render_video_path = "debug_output/taijiquan.mp4"
npz_path = "data/demo_npzs/taijiquan_female_ID0_difTraj_raw.npz"
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


