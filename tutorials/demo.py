import torch
import numpy as np
from os import path as osp

from human_body_prior.model_hub import get_vposer_model

device = "cuda"

vp = get_vposer_model(device=device, vposer_ckpt="data/vposer_v02_05")

sample_amass_fname = "support_data/dowloads/amass_sample.npz"

# Prepare the pose_body from amass sample
amass_body_pose = np.load(sample_amass_fname)['poses'][:, 3:66]
amass_body_pose = torch.from_numpy(amass_body_pose).type(torch.float).to('cuda')
print('amass_body_pose.shape', amass_body_pose.shape)

amass_body_poZ = vp.encode(amass_body_pose).mean
print('amass_body_poZ.shape', amass_body_poZ.shape)

