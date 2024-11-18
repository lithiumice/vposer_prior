# Visualize
import os
# export PYOPENGL_PLATFORM=osmesa
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import show_image
#Loading SMPLx Body Model
from human_body_prior.body_model.body_model import BodyModel
bm = BodyModel(bm_fname="/data/motion/body_models/smplx/SMPLX_NEUTRAL.npz").to('cuda')

from misc_utils import *


all_files = glob(f"debug/tram_fit_data_0804_part/*.npz")
npz_files_2 = random.sample(all_files, 20)

frame_idx = 0
poses = extract_poses(npz_files_2[0]).cuda()
images = render_smpl_params(bm, {'pose_body':poses[frame_idx:frame_idx+1]}, cam_trans=[0, -0.5, 2.0], mesh_color="blue")[None,None,...]

