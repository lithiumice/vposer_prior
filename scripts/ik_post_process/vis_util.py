import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# Set matplotlib backend for headless environment
import matplotlib
matplotlib.use('Agg')



def visualize_skeleton_animation(skeleton, skeleton_connections=None, save_animation=True, save_frames=False, contacts=None, filename_suffix=''):
    """
    Visualize skeleton animation using matplotlib
    
    Args:
        skeleton: numpy array of shape (T, J, 3) where T is frames, J is joints
        skeleton_connections: list of tuples defining which joints to connect
        save_animation: whether to save animation as GIF
        save_frames: whether to save individual frames as images
        contacts: dictionary with ground contact information
    """
    T, J, _ = skeleton.shape
    
    from const import SMPLX_skeleton_connections
    skeleton_connections = SMPLX_skeleton_connections
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Skeleton Animation')
    
    # Set axis limits based on skeleton bounds
    margin = 0.1
    x_min, x_max = skeleton[:, :, 0].min(), skeleton[:, :, 0].max()
    y_min, y_max = skeleton[:, :, 1].min(), skeleton[:, :, 1].max()
    z_min, z_max = skeleton[:, :, 2].min(), skeleton[:, :, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    # Initialize plot elements
    lines = []
    points = ax.scatter([], [], [], c='red', s=50)
    
    # Initialize contact points
    contact_points = None
    if contacts:
        contact_points = ax.scatter([], [], [], c='lime', s=100, marker='o')
    
    # Create line objects for connections
    for _ in skeleton_connections:
        line, = ax.plot([], [], [], 'b-', linewidth=2)
        lines.append(line)
    
    def animate(frame):
        # Update point positions
        points._offsets3d = (skeleton[frame, :, 0], 
                           skeleton[frame, :, 1], 
                           skeleton[frame, :, 2])
        
        # Update contact points
        if contacts and contact_points:
            contact_x, contact_y, contact_z = [], [], []
            for joint_name, contact_info in contacts.items():
                if contact_info['is_contact'][frame]:
                    joint_idx = contact_info['joint_idx']
                    contact_x.append(skeleton[frame, joint_idx, 0])
                    contact_y.append(skeleton[frame, joint_idx, 1])
                    contact_z.append(skeleton[frame, joint_idx, 2])
            
            if contact_x:
                contact_points._offsets3d = (contact_x, contact_y, contact_z)
            else:
                contact_points._offsets3d = ([], [], [])
        
        # Update line connections
        for i, (start_idx, end_idx) in enumerate(skeleton_connections):
            if start_idx < J and end_idx < J:
                x_data = [skeleton[frame, start_idx, 0], skeleton[frame, end_idx, 0]]
                y_data = [skeleton[frame, start_idx, 1], skeleton[frame, end_idx, 1]]
                z_data = [skeleton[frame, start_idx, 2], skeleton[frame, end_idx, 2]]
                lines[i].set_data_3d(x_data, y_data, z_data)
        
        if contact_points:
            return [points, contact_points] + lines
        else:
            return [points] + lines
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=T, interval=50, blit=False, repeat=True)
    
    # Save animation as GIF
    if save_animation:
        filename = f'{filename_suffix}.gif'
        print(f"Saving animation as {filename}...")
        writer = PillowWriter(fps=20)
        anim.save(filename, writer=writer)
        print("Animation saved!")
    
    # # Save individual frames
    # if save_frames:
    #     frame_dir = f'skeleton_frames{filename_suffix}'
    #     print("Saving individual frames...")
    #     os.makedirs(frame_dir, exist_ok=True)
    #     for i in range(0, min(T, 50), 5):  # Save every 5th frame, max 10 frames
    #         animate(i)
    #         plt.savefig(f'{frame_dir}/frame_{i:03d}.png', dpi=100, bbox_inches='tight')
    #     print(f"Frames saved to {frame_dir}/ directory")
    
    plt.close()
    
    return anim

def visualize_static_frames(skeleton, num_frames=6, skeleton_connections=None, contacts=None, filename_suffix=''):
    T, J, _ = skeleton.shape
    
    # Default SMPLX skeleton connections (simplified)
    if skeleton_connections is None:
        skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # spine chain
            (1, 5), (5, 6), (6, 7),          # left arm
            (1, 8), (8, 9), (9, 10),         # right arm
            (0, 11), (11, 12), (12, 13),    # left leg
            (0, 14), (14, 15), (15, 16),    # right leg
        ]
    
    # Select frames to display
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    # Create subplots
    fig = plt.figure(figsize=(15, 10))
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        # Plot joints
        ax.scatter(skeleton[frame_idx, :, 0], 
                  skeleton[frame_idx, :, 1], 
                  skeleton[frame_idx, :, 2], 
                  c='red', s=30)
        
        # Plot contact points with different colors
        if contacts:
            for joint_name, contact_info in contacts.items():
                if contact_info['is_contact'][frame_idx]:
                    joint_idx = contact_info['joint_idx']
                    ax.scatter(skeleton[frame_idx, joint_idx, 0], 
                              skeleton[frame_idx, joint_idx, 1], 
                              skeleton[frame_idx, joint_idx, 2], 
                              c='lime', s=100, marker='o')
        
        # Plot connections
        for start_idx, end_idx in skeleton_connections:
            if start_idx < J and end_idx < J:
                x_data = [skeleton[frame_idx, start_idx, 0], skeleton[frame_idx, end_idx, 0]]
                y_data = [skeleton[frame_idx, start_idx, 1], skeleton[frame_idx, end_idx, 1]]
                z_data = [skeleton[frame_idx, start_idx, 2], skeleton[frame_idx, end_idx, 2]]
                ax.plot(x_data, y_data, z_data, 'b-', linewidth=1)
        
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set consistent axis limits
        margin = 0.1
        x_min, x_max = skeleton[:, :, 0].min(), skeleton[:, :, 0].max()
        y_min, y_max = skeleton[:, :, 1].min(), skeleton[:, :, 1].max()
        z_min, z_max = skeleton[:, :, 2].min(), skeleton[:, :, 2].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    plt.tight_layout()
    filename = f'skeleton_frames_grid{filename_suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Static frames visualization saved as {filename}")
    plt.close()
