# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import numpy as np
from collections import namedtuple


from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints, blend_shapes, batch_rigid_transform, batch_rodrigues


from .smpl_constants import *

ModelOutput = namedtuple('ModelOutput',
                         ['vertices',
                          'joints', 'full_pose', 'betas',
                          'global_orient',
                          'body_pose', 'expression',
                          'left_hand_pose', 'right_hand_pose',
                          'jaw_pose',
                          'global_trans',
                          'scale'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


SMPL_MODEL_DIR = "/data/motion/body_models/smpl/"
_DATA = "data/glamr_data"


JOINT_REGRESSOR_TRAIN_EXTRA = f'{_DATA}/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = f'{_DATA}/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = f'{_DATA}/smpl_mean_params.npz'



class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        super(SMPL, self).__init__(*args, **kwargs, pose2rot = 'aa')
        if 'pose_type' in kwargs.keys():
            self.joint_names = get_ordered_joint_names(kwargs['pose_type'] )
        else:
            self.joint_names = JOINT_NAMES

        joints = [JOINT_MAP[i] for i in self.joint_names]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, root_trans=None, root_scale=None, orig_joints=False, **kwargs):
        """
        root_trans: B x 3, root translation
        root_scale: B, scale factor w.r.t root
        """
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs, pose2rot = 'aa')
        if orig_joints:
            joints = smpl_output.joints[:, :24]
        else:
            extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
            joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
            joints = joints[:, self.joint_map, :]

        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        if root_trans is not None:
            if root_scale is None:
                root_scale = torch.ones_like(root_trans[:, 0])
            cur_root_trans = joints[:, [0], :]
            # rel_trans = (root_trans - joints[:, 0, :]).unsqueeze(1)
            output.vertices[:] = (output.vertices - cur_root_trans) * root_scale[:, None, None] + root_trans[:, None, :]
            output.joints[:] = (output.joints - cur_root_trans) * root_scale[:, None, None] + root_trans[:, None, :]
        return output

    def get_joints(self, betas=None, body_pose=None, global_orient=None, transl=None,
                   pose2rot=True, root_trans=None, root_scale=None, dtype=torch.float32):
        # If no shape and lib parameters are passed along, then use the
        # ones from the module

        pose = torch.cat([global_orient, body_pose], dim=1)

        """ LBS """
        batch_size = pose.shape[0]
        J = torch.matmul(self.J_regressor, self.v_template).repeat((batch_size, 1, 1))
        if pose2rot:
            rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        else:
            rot_mats = pose.view(batch_size, -1, 3, 3)
        joints, A = batch_rigid_transform(rot_mats, J, self.parents, dtype=torch.float32)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)

        if root_trans is not None:
            if root_scale is None:
                root_scale = torch.ones_like(root_trans[:, 0])
            cur_root_trans = joints[:, [0], :]
            joints[:] = (joints - cur_root_trans) * root_scale[:, None, None] + root_trans[:, None, :]

        return joints
        
        

def get_smpl_faces():
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces

