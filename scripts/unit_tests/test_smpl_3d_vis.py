
from body_visualizer.vis3d.visualizer3d_smpl import SMPLVisualizer
import numpy as np
import torch

if __name__ == "__main__":
    visualizer = SMPLVisualizer(generator_func=None, distance=7, device="cuda", enable_shadow=True, anti_aliasing=True,
                                smpl_model_dir="data/smpl", sample_visible_alltime=True, verbose=True, enable_cam_following=False)

    # import ipdb; ipdb.set_trace()
    npz_path = "data/demo_npzs/taijiquan_female_ID0_difTraj_raw.npz"
    data = np.load(npz_path)
    
    # shape: num_persion, T, ...
    common_op = lambda x: x[:150].unsqueeze(0)

    smpl_seq = {
        'pose': common_op(torch.tensor(data['poses'])[:,:24,:].reshape(-1,72)),
        "trans": common_op(torch.tensor(data["trans"])),
        "shape": common_op(torch.tensor(data["betas"])),
    }
    
    init_args = {'smpl_seq': smpl_seq, 'mode': 'gt'}
    
    save_video_path = "./debug/vis_3d/test2.mp4"
    visualizer.save_animation_as_video(save_video_path, init_args=init_args, window_size=(1500, 1500), cleanup=True)
    print(f"[INFO] Save video to {save_video_path}")
    
    # visualizer.show_animation(window_size=(800, 800), init_args=init_args, enable_shadow=None, frame_mode='fps', fps=30, repeat=False, show_axes=True)
    