
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




@ray.remote(num_gpus=0.3)
def process_imitation(npz_path, render_video_path):
    
    visualizer = SMPLVisualizer(generator_func=None, distance=7, device="cuda", enable_shadow=True, anti_aliasing=True,
                                smpl_model_dir="data/smpl", sample_visible_alltime=True, verbose=True, enable_cam_following=False)
    
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
    
    
def run(data_params, process_params, model_paths, model_params, scaling_params):

    ########### init ray cuda device ##############
    if (visible_gpus := config.scaling_params.CUDA_VISIBLE_DEVICES) is not None:
        # set os environment so torch can only see these GPU devices
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        
        # used for ray only
        gpus = list(map(int, visible_gpus.split(",")))
        num_gpus = len(gpus)
        print(f"[INFO] num_gpus: {num_gpus}, gpus: {gpus}")
    else:
        print(f"[INFO] CUDA_VISIBLE_DEVICES is not set, using all GPUs")
        num_gpus = torch.cuda.device_count()
        
        
    npz_path_list = glob(data_params["glob_pattern"])

    if process_params.get("debug", False):
        npz_path_list = npz_path_list[:10]

    tasks = []
    for npz_path in npz_path_list:
        output_folder = data_params["output_folder"]
        render_video_path = str(Path(output_folder, Path(npz_path).stem + ".mp4"))
        
        if Path(render_video_path).exists():
            # print(f"[DEBUG] {record_npz_path} exist, skip")
            continue
        
        # on RTX 3060, process one npz take up < 4GB GPU memory
        tasks.append(process_imitation.remote(npz_path, render_video_path))

    ray.get(tasks)
    ray.shutdown()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="scripts/ray_cluster/configs/local_rtx3060_infer.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    run(
        data_params=config["data"],
        process_params=config["process_params"],
        model_paths=config["model_paths"],
        model_params=config["model_params"],
        scaling_params=config["scaling_params"],
    )
