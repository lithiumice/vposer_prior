
from body_visualizer.vis3d.vis_smpl import SMPLVisualizer
import numpy as np
import torch

if __name__ == "__main__":
    visualizer = SMPLVisualizer(generator_func=None, distance=7, device="cuda", 
                                smpl_model_dir="data/smpl",
                                sample_visible_alltime=True, verbose=False)

    # import ipdb; ipdb.set_trace()
    npz_path = "data/demo_npzs/taijiquan_female_ID0_difTraj_raw.npz"
    data = np.load(npz_path)

    smpl_seq = {
        'pose': (torch.tensor(data['poses'])[:,:24,:].reshape(-1,72)),
        "trans": (torch.tensor(data["trans"])),
        "shape": (torch.tensor(data["betas"]).unsqueeze(0)),
    }

    visualizer.save_animation_as_video("test", init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1500, 1500), cleanup=True)
    
    