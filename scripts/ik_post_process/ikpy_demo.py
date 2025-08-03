
from ikpy.chain import Chain
from ikpy.link import URDFLink
IKPY_AVAILABLE = True


def create_smplx_leg_chains():
    """
    Create IK chains for SMPLX leg joints
    Returns left and right leg chains from pelvis to foot
    """
    if not IKPY_AVAILABLE:
        return None, None
    
    # SMPLX joint indices for leg chains
    # pelvis (0) -> left_hip (1) -> left_knee (4) -> left_ankle (7) -> left_foot (10)
    left_leg_chain = Chain(name='left_leg', links=[
        URDFLink(name="pelvis", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="left_hip", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="left_knee", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="left_ankle", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="left_foot", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
    ])
    
    # pelvis (0) -> right_hip (2) -> right_knee (5) -> right_ankle (8) -> right_foot (11)
    right_leg_chain = Chain(name='right_leg', links=[
        URDFLink(name="pelvis", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="right_hip", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="right_knee", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="right_ankle", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
        URDFLink(name="right_foot", translation_vector=[0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 0], bounds="unbounded"),
    ])
    
    return left_leg_chain, right_leg_chain

def apply_ik_correction(skeleton, contacts, max_iter=10):
    """
    Apply IK correction to eliminate foot sliding for detected ground contacts
    
    Args:
        skeleton: numpy array of shape (T, J, 3)
        contacts: dictionary with ground contact information
        max_iter: maximum iterations for IK solver
    
    Returns:
        corrected_skeleton: numpy array with IK-corrected joint positions
    """
    if not IKPY_AVAILABLE:
        print("IK correction skipped: ikpy not available")
        return skeleton
    
    T, J, _ = skeleton.shape
    corrected_skeleton = skeleton.copy()
    
    # Create IK chains
    left_leg_chain, right_leg_chain = create_smplx_leg_chains()
    
    # Get joint indices
    joint_indices = {
        'left_foot': SMPLX_JOINT_NAMES.index('left_foot'),
        'right_foot': SMPLX_JOINT_NAMES.index('right_foot'),
        'left_ankle': SMPLX_JOINT_NAMES.index('left_ankle'),
        'right_ankle': SMPLX_JOINT_NAMES.index('right_ankle'),
        'left_hip': SMPLX_JOINT_NAMES.index('left_hip'),
        'right_hip': SMPLX_JOINT_NAMES.index('right_hip'),
        'left_knee': SMPLX_JOINT_NAMES.index('left_knee'),
        'right_knee': SMPLX_JOINT_NAMES.index('right_knee'),
        'pelvis': SMPLX_JOINT_NAMES.index('pelvis')
    }
    
    # Find ground level
    ground_level = np.min(skeleton[:, :, 2])
    
    # Process each frame
    for frame in range(1, T):  # Start from frame 1 to compare with previous
        # Check which joints are in contact
        contact_joints = []
        target_positions = []
        
        for joint_name in ['left_foot', 'right_foot']:
            if contacts[joint_name]['is_contact'][frame]:
                contact_joints.append(joint_name)
                # Use previous frame position as target to eliminate sliding
                prev_pos = skeleton[frame-1, joint_indices[joint_name]]
                # Keep the same x,y position but lock to ground level
                target_positions.append([prev_pos[0], prev_pos[1], ground_level])
        
        # Apply IK correction for each contact joint
        for joint_name, target_pos in zip(contact_joints, target_positions):
            if 'left' in joint_name:
                chain = left_leg_chain
                hip_idx = joint_indices['left_hip']
                knee_idx = joint_indices['left_knee']
                ankle_idx = joint_indices['left_ankle']
                foot_idx = joint_indices['left_foot']
            else:
                chain = right_leg_chain
                hip_idx = joint_indices['right_hip']
                knee_idx = joint_indices['right_knee']
                ankle_idx = joint_indices['right_ankle']
                foot_idx = joint_indices['right_foot']
            
            # Get current joint positions
            pelvis_pos = corrected_skeleton[frame, joint_indices['pelvis']]
            hip_pos = corrected_skeleton[frame, hip_idx]
            knee_pos = corrected_skeleton[frame, knee_idx]
            ankle_pos = corrected_skeleton[frame, ankle_idx]
            foot_pos = corrected_skeleton[frame, foot_idx]
            
            # Simple IK correction - adjust ankle and foot positions
            # This is a simplified approach that maintains the leg structure
            # while eliminating foot sliding
            
            # Calculate the vector from hip to target
            hip_to_target = np.array(target_pos) - hip_pos
            hip_to_target[2] = 0  # Project to horizontal plane
            
            # Calculate the vector from hip to current foot
            hip_to_foot = foot_pos - hip_pos
            hip_to_foot[2] = 0  # Project to horizontal plane
            
            # Calculate the correction needed
            if np.linalg.norm(hip_to_foot) > 0:
                correction = hip_to_target - hip_to_foot
                
                # Apply correction to ankle and foot
                corrected_skeleton[frame, ankle_idx] += correction * 0.7
                corrected_skeleton[frame, foot_idx] += correction
                
                # Slight adjustment to knee to maintain leg structure
                corrected_skeleton[frame, knee_idx] += correction * 0.3
    
    return corrected_skeleton



# Apply IK correction to eliminate foot sliding
print("\nApplying IK correction to eliminate foot sliding...")
corrected_skeleton = apply_ik_correction(skeleton, contacts, max_iter=10)

# Re-detect contacts on corrected skeleton
corrected_contacts = detect_ground_contacts(corrected_skeleton, height_threshold=0.1, velocity_threshold=0.2)

print("\nCorrected ground contact statistics:")
for joint_name, contact_info in corrected_contacts.items():
    contact_frames = np.sum(contact_info['is_contact'])
    contact_percentage = (contact_frames / len(corrected_skeleton)) * 100
    print(f"  {joint_name}: {contact_frames}/{len(corrected_skeleton)} frames ({contact_percentage:.1f}%)")

# Save both original and corrected skeleton data
np.savez('skeleton_output.npz', 
            skeleton=skeleton,
            corrected_skeleton=corrected_skeleton,
            poses=motion_data['poses'],
            betas=motion_data['betas'],
            trans=motion_data['trans'])

print(f"\nSkeleton data saved to skeleton_output.npz")
print(f"First frame joint positions range: [{np.min(skeleton[0]):.3f}, {np.max(skeleton[0]):.3f}]")

# Visualize the original skeleton animation with contact detection
print("\nGenerating original skeleton animation visualization...")
anim = visualize_skeleton_animation(skeleton, save_animation=True, save_frames=True, contacts=contacts)

# Visualize the corrected skeleton animation
print("\nGenerating corrected skeleton animation visualization...")
corrected_anim = visualize_skeleton_animation(corrected_skeleton, save_animation=True, save_frames=False, contacts=corrected_contacts, filename_suffix='_corrected')

# Also create static frames visualization with contact detection
print("\nGenerating static frames visualization...")
visualize_static_frames(skeleton, num_frames=6, contacts=contacts)

# Create static frames for corrected skeleton
print("\nGenerating corrected static frames visualization...")
visualize_static_frames(corrected_skeleton, num_frames=6, contacts=corrected_contacts, filename_suffix='_corrected')

print("\nVisualization complete! Check the following files:")
print("- skeleton_animation.gif: Original animated skeleton with contact points")
print("- skeleton_animation_corrected.gif: IK-corrected animated skeleton")
print("- skeleton_frames_grid.png: Original static grid of frames with contact points")
print("- skeleton_frames_grid_corrected.png: IK-corrected static grid")
print("- skeleton_frames/: Individual frame images")
print("\nContact points are shown in lime green (larger circles)")
print("IK correction reduces foot sliding by locking contact joints to previous positions")