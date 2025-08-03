import numpy as np
import torch
import smplx
from const import SMPLX_JOINT_NAMES


def load_motion_and_get_skeleton():
    # Load motion data
    motion_data = np.load('data/demo_npzs/taijiquan_female_ID0_difTraj_raw.npz')
    
    print("Motion data keys and shapes:")
    for key in motion_data.files:
        print(f"  {key}: {motion_data[key].shape}")
    
    # Extract motion parameters
    poses = motion_data['poses']  # (T, 55, 3)
    betas = motion_data['betas']  # (10,)
    trans = motion_data['trans']  # (T, 3)
    gender = motion_data['gender']  # 'male' or 'female'
    
    T = poses.shape[0]  # Number of frames
    print(f"\nMotion data: {T} frames, {poses.shape[1]} joints")
    
    # Load SMPLX model
    smplx_model = smplx.create(
        model_path='data',
        model_type='smplx',
        gender='neutral',
        use_face_contour=False,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz',
        flat_hand_mean=True
    )
    
    # Convert to tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smplx_model = smplx_model.to(device)
    
    poses_tensor = torch.tensor(poses, dtype=torch.float32, device=device)
    betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)
    trans_tensor = torch.tensor(trans, dtype=torch.float32, device=device)
    
    # Run SMPLX forward pass for each frame
    skeleton_joints = []
    
    with torch.no_grad():
        for i in range(T):
            # SMPLX expects poses in (N, 165) format where N is batch size
            # Each pose has 55 joints * 3 = 165 parameters
            pose_frame = poses_tensor[i].reshape(1, -1)  # (1, 165)
            trans_frame = trans_tensor[i].reshape(1, -1)  # (1, 3)
            
            # Forward pass through SMPLX
            output = smplx_model(
                body_pose=pose_frame[:, 3:66],  # Body joints (excluding pelvis)
                global_orient=pose_frame[:, :3],  # Global orientation
                betas=betas_tensor,
                transl=trans_frame
            )
            
            # Get joint positions (J, 3)
            joints = output.joints.detach().cpu().numpy()[0]  # Remove batch dimension
            skeleton_joints.append(joints)
    
    # Convert to numpy array (T, J, 3)
    skeleton = np.stack(skeleton_joints, axis=0)
    
    print(f"\nSkeleton shape: {skeleton.shape}")
    print(f"Number of joints: {skeleton.shape[1]}")
    print(f"Skeleton data type: {skeleton.dtype}")

    # post_w_j3d = torch.from_numpy(skeleton)[:, :22, :]
    # local_mat = axis_angle_to_matrix(poses_tensor)[:, :22]

    return skeleton, motion_data

def detect_ground_contacts(skeleton, height_threshold=0.05, velocity_threshold=0.1):
    """
    Detect ground contacts for foot and ankle joints using heuristic approach
    
    Args:
        skeleton: numpy array of shape (T, J, 3) where T is frames, J is joints
        height_threshold: height threshold for ground contact (meters)
        velocity_threshold: velocity threshold for ground contact (m/s)
    
    Returns:
        contacts: dictionary with contact information for each joint
    """
    T, J, _ = skeleton.shape
    
    # Joint indices for feet and ankles
    joint_indices = {
        'left_foot': SMPLX_JOINT_NAMES.index('left_foot'),
        'right_foot': SMPLX_JOINT_NAMES.index('right_foot'),
        'left_ankle': SMPLX_JOINT_NAMES.index('left_ankle'),
        'right_ankle': SMPLX_JOINT_NAMES.index('right_ankle')
    }
    
    # Find ground level (minimum z-coordinate across all frames and joints)
    ground_level = np.min(skeleton[:, :, 2])
    
    contacts = {}
    
    for joint_name, joint_idx in joint_indices.items():
        # Get joint positions
        joint_positions = skeleton[:, joint_idx, :]
        
        # Calculate height above ground
        heights = joint_positions[:, 2] - ground_level
        
        # Calculate velocity (magnitude of 3D velocity)
        velocities = np.zeros(T)
        velocities[1:] = np.linalg.norm(joint_positions[1:] - joint_positions[:-1], axis=1)
        velocities[0] = velocities[1]  # Copy first velocity
        
        # Detect contacts (height < threshold AND velocity < threshold)
        is_contact = (heights < height_threshold) & (velocities < velocity_threshold)
        
        contacts[joint_name] = {
            'is_contact': is_contact,
            'heights': heights,
            'velocities': velocities,
            'joint_idx': joint_idx
        }
    
    return contacts



import matrix
from ccd_ik import CCD_IK
from pytorch3d.transforms import *
from smplx_lite import SmplxLiteV437Coco17
smplx_model = SmplxLiteV437Coco17()
parents = smplx_model.parents[:22]
parents_tensor = torch.tensor(parents)
parents = parents.tolist()



def fk_v2(body_pose, betas, global_orient=None, transl=None):
    """
    Args:
        body_pose: (B, L, 63)
        betas: (B, L, 10)
        global_orient: (B, L, 3)
    Returns:
        joints: (B, L, 22, 3)
    """
    B, L = body_pose.shape[:2]
    if global_orient is None:
        global_orient = torch.zeros((B, L, 3), device=body_pose.device)
    aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
    rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

    skeleton = smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
    local_skeleton = skeleton - skeleton[:, :, parents_tensor]
    local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

    if transl is not None:
        local_skeleton[..., 0, :] += transl  # B, L, 22, 3

    mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
    fk_mat = matrix.forward_kinematics(mat, parents)  # B, L, 22, 4, 4
    joints = matrix.get_position(fk_mat)  # B, L, 22, 3
    return joints, mat, fk_mat
    
    
    
if __name__ == "__main__":
    skeleton, motion_data = load_motion_and_get_skeleton()
    
    # Detect ground contacts
    print("\nDetecting ground contacts...")
    contacts = detect_ground_contacts(skeleton, height_threshold=0.1, velocity_threshold=0.2)
    
    # Print contact statistics
    print("\nGround contact statistics:")
    for joint_name, contact_info in contacts.items():
        contact_frames = np.sum(contact_info['is_contact'])
        contact_percentage = (contact_frames / len(skeleton)) * 100
        print(f"  {joint_name}: {contact_frames}/{len(skeleton)} frames ({contact_percentage:.1f}%)")
    
        
    ############## IK
    import ipdb; ipdb.set_trace()
    body_pose = motion_data['poses'][:, 3:]  # Exclude global orientation
    betas = motion_data['betas']  # (10,)
    global_orient= motion_data['poses'][:, :3]  # Global orientation
    transl = motion_data['trans']  # Translation
    
    static_conf = contacts  # (B, L, J)
    post_w_j3d, local_mat, post_w_mat = fk_v2(body_pose, betas, global_orient, transl)

    # sebas rollout merge
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    # ik
    global_rot = matrix.get_rotation(post_w_mat)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        IK_solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )

        chain_local_mat = IK_solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        return local_mat

    local_mat = ik(local_mat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))  # (B, L, J-1, 3, 3)
    body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

    