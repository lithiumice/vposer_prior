import torch
import numpy as np
from collections import namedtuple


# H36 constants
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
H36M_TO_J15 = [H36M_TO_J17[14]] + H36M_TO_J17[:14]
H36M_TO_J16 = H36M_TO_J17[14:16] + H36M_TO_J17[:14]


# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27,
    'Left Thumb Tip': 35, 'Left Index Tip': 36, 'Left Middle Tip': 37,
    'Left Ring Tip': 38, 'Left Pinky Tip': 39,
    'Right Thumb Tip': 40, 'Right Index Tip': 41, 'Right Middle Tip': 42,
    'Right Ring Tip': 43, 'Right Pinky Tip': 44
}

JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

SMPL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'right_index1'
]


def print_smpl_joint_val(x_dict, include_root=False):
    sind = 0 if include_root else 1
    jnum = 24 - sind
    for i in range(jnum):
        jstr = SMPL_JOINT_NAMES[i + sind]
        val_str = ' --- '.join([name + ': ' + np.array2string(x[..., 3 * i: 3* (i + 1)], precision=3, suppress_small=True, sign=' ') for name, x in x_dict.items()])
        print(f' {jstr:20} --- {val_str}')
    return



def get_ordered_joint_names(pose_type):
    joint_names = None
    if pose_type == 'body26':
        # joints order according to
        joint_names = [
            'Pelvis (MPII)',  # 0
            'OP LHip',  # 1
            'OP RHip',  # 2
            'Spine (H36M)',  # 3
            'OP LKnee',  # 4
            'OP RKnee',  # 5
            'OP Neck',  # 6
            'OP LAnkle',  # 7
            'OP RAnkle',  # 8
            'OP LBigToe',  # 9
            'OP RBigToe',  # 10
            'OP LSmallToe',  # 11
            'OP RSmallToe',  # 12
            'OP LHeel',  # 13
            'OP RHeel',  # 14
            'OP Nose',  # 15
            'OP LEye',  # 16
            'OP REye',  # 17
            'OP LEar',  # 18
            'OP REar',  # 19
            'OP LShoulder',  # 20
            'OP RShoulder',  # 21
            'OP LElbow',  # 22
            'OP RElbow',  # 23
            'OP LWrist',  # 24
            'OP RWrist',  # 25
        ]
    elif pose_type == 'body34':
        joint_names = [
            'Pelvis (MPII)',  # 0
            'OP LHip',  # 1
            'OP RHip',  # 2
            'Spine (H36M)',  # 3
            'OP LKnee',  # 4
            'OP RKnee',  # 5
            'OP Neck',  # 6
            'OP LAnkle',  # 7
            'OP RAnkle',  # 8
            'OP LBigToe',  # 9
            'OP RBigToe',  # 10
            'OP LSmallToe',  # 11
            'OP RSmallToe',  # 12
            'OP LHeel',  # 13
            'OP RHeel',  # 14
            'OP Nose',  # 15
            'OP LEye',  # 16
            'OP REye',  # 17
            'OP LEar',  # 18
            'OP REar',  # 19
            'OP LShoulder',  # 20
            'OP RShoulder',  # 21
            'OP LElbow',  # 22
            'OP RElbow',  # 23
            'OP LWrist',  # 24
            'OP RWrist',  # 25
            'Left Pinky Tip', # 26 FIXME Using Tip instead of Knuckle.
            'Right Pinky Tip', # 27 FIXME Using Tip instead of Knuckle.
            'Left Middle Tip', #28
            'Right Middle Tip', #29
            'Left Index Tip',  # 30 FIXME: Using Tip instead of knuckle
            'Right Index Tip', # 31 FIXME: Using Tip instead of knuckle
            'Left Thumb Tip', # 32
            'Right Thumb Tip' #33
         ]
    elif pose_type == 'body30':
        joint_names = [
            'Pelvis (MPII)',  # 0
            'OP LHip',  # 1
            'OP RHip',  # 2
            'Spine (H36M)',  # 3
            'OP LKnee',  # 4
            'OP RKnee',  # 5
            'OP Neck',  # 6
            'OP LAnkle',  # 7
            'OP RAnkle',  # 8
            'OP LBigToe',  # 9
            'OP RBigToe',  # 10
            'OP LSmallToe',  # 11
            'OP RSmallToe',  # 12
            'OP LHeel',  # 13
            'OP RHeel',  # 14
            'OP Nose',  # 15
            'OP LEye',  # 16
            'OP REye',  # 17
            'OP LEar',  # 18
            'OP REar',  # 19
            'OP LShoulder',  # 20
            'OP RShoulder',  # 21
            'OP LElbow',  # 22
            'OP RElbow',  # 23
            'OP LWrist',  # 24
            'OP RWrist',  # 25
            'Left Pinky Tip', # 26 FIXME Using Tip instead of Knuckle.
            'Right Pinky Tip', # 27 FIXME Using Tip instead of Knuckle.
            'Left Index Tip',  # 30 FIXME: Using Tip instead of knuckle
            'Right Index Tip', # 31 FIXME: Using Tip instead of knuckle
         ]

    elif pose_type == "body26fk":
        joint_names = [
            #'Pelvis (MPII)',  # 0
            'Pelvis (MPII)',  # 0
            'OP LHip',  # 1
            'OP RHip',  # 2
            'Spine (H36M)',  # 3
            'OP LKnee',  # 4
            'OP RKnee',  # 5
            'OP Neck',  # 6
            'OP LAnkle',  # 7
            'OP RAnkle',  # 8
            'OP LBigToe',  # 9
            'OP RBigToe',  # 10
            'OP LSmallToe',  # 11
            'OP RSmallToe',  # 12
            'OP LHeel',  # 13
            'OP RHeel',  # 14
            'OP Nose',  # 15
            'OP LEye',  # 16
            'OP REye',  # 17
            'OP LEar',  # 18
            'OP REar',  # 19
            'OP LShoulder',  # 20
            'OP RShoulder',  # 21
            'OP LElbow',  # 22
            'OP RElbow',  # 23
            'OP LWrist',  # 24
            'OP RWrist',  # 25
        ]
    elif pose_type == "body15":

        joint_names = [
            'Pelvis (MPII)',  # 0
            'OP RAnkle',  # 1
            'OP RKnee',  # 2
            'OP RHip',  # 3
            'OP LHip',  # 4
            'OP LKnee',  # 5
            'OP LAnkle',  # 6
            'OP RWrist',  # 7
            'OP RElbow',  # 8
            'OP RShoulder',  # 9
            'OP LShoulder',  # 10
            'OP LElbow',  # 11
            'OP LWrist',  # 12
            'Neck (LSP)',  # 13
            'Top of Head (LSP)'  # 14
        ]

    return joint_names
