# import sys
# import os
# import argparse
# import numpy as np
# import os
# import torch
# import cv2
# from tqdm import tqdm
# from glob import glob
# from pathlib import Path
# import imageio

# from body_visualizer.vis3d.renderer_pytorch3d_standalone import Renderer
# from body_visualizer.vis3d.smpl import SMPL


# # global constants
# device = torch.device('cuda')

# smpl = SMPL("data/smpl", pose_type='body26fk', create_transl=False).to(device)


# colors = np.loadtxt('data/colors.txt')/255
# colors = torch.from_numpy(colors).float()



# pred = self.smpl(
#                 global_orient=pose[..., :3].view(-1, 3),
#                 body_pose=pose[..., 3:].view(-1, 69),
#                 root_trans=trans.view(-1, 3),
#                 betas=shape.view(-1, 10),
#                 return_full_pose=True,
#                 orig_joints=True
#             )
            
# pred_vert_w = pred.vertices
# pred_j3d_w = pred.joints[:, :24]

# locations.append(pred_j3d_w.mean(1))

# for j, f in enumerate(frame.tolist()):
#     track_tid[f].append(i)
#     track_verts[f].append(pred_vert_w[j])
#     track_joints[f].append(pred_j3d_w[j])




# ##### Fit to Ground #####
# grounding_verts = []
# grounding_joints = []
# frames_wise_verts_flag = [t for t in tstamp if (len(track_verts[t])!=0)]
# for t in frames_wise_verts_flag[:20]:
#     verts = torch.stack(track_verts[t])
#     joints = torch.stack(track_joints[t])
#     grounding_verts.append(verts)
#     grounding_joints.append(joints)
    
# grounding_verts = torch.cat(grounding_verts)
# grounding_joints = torch.cat(grounding_joints)

# print(f"[DEBUG] {grounding_verts.dtype=}")
# R, offset = fit_to_ground_easy(grounding_verts, grounding_joints)
# offset = totype(torch.tensor([0, offset, 0]))

# locations = torch.cat(locations)
# locations = torch.einsum('ij,bj->bi', R, locations) - offset
# cx, cz = (locations.max(0)[0] + locations.min(0)[0])[[0, 2]] / 2.0
# sx, sz = (locations.max(0)[0] - locations.min(0)[0])[[0, 2]]
# scale = max(sx.item(), sz.item()) * 2

# ##### Viewing Camera #####
# pred_camt = totype(torch.tensor(pred_traj[:, :3]) * pred_cam['scale']) 
# pred_camq = totype(torch.tensor(pred_traj[:, 3:]))
# pred_camr = quaternion_to_matrix(pred_camq[:,[3,0,1,2]])

# cam_R = torch.einsum('ij,bjk->bik', R, pred_camr)
# cam_T = torch.einsum('ij,bj->bi', R, pred_camt) - offset
# cam_R = cam_R.mT
# cam_T = - torch.einsum('bij,bj->bi', cam_R, cam_T)

# cam_R = cam_R.to(device)
# cam_T = cam_T.to(device)

# ##### Render video for visualization #####
# writer = imageio.get_writer(render_video_save_path := f'{seq_folder}/tram_output.mp4', fps=30, mode='I', 
#                             format='FFMPEG', macro_block_size=1)
# bin_size = 128 # rendering takes 5.5it/s
# max_faces_per_bin = 20000

# img_focal=5000
# h=w=800
# renderer = Renderer(h, w, img_focal-100, device, 
#                     smpl.faces, bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)
# renderer.set_ground(scale, cx.item(), cz.item())

# for i in tqdm(range(len(vr))):
#     # img = cv2.imread(imgfiles[i])[:,:,::-1]
    
#     verts_list = track_verts[i]
#     if len(verts_list)>0:
#         verts_list = torch.stack(track_verts[i])#[:,None]
#         verts_list = torch.einsum('ij,bnj->bni', R, verts_list)[:,None]
#         verts_list -= offset
#         verts_list = verts_list.to(device)
        
#         tid = track_tid[i]
#         verts_colors = torch.stack([colors[t] for t in tid]).to(device)

#     faces = renderer.faces.clone().squeeze(0)
#     cameras, lights = renderer.create_camera_from_cv(cam_R[[i]], cam_T[[i]])
#     rend = renderer.render_with_ground_multiple(verts_list, faces, verts_colors, cameras, lights)
    
#     writer.append_data(out)

# writer.close()
# print(f'[INFO] Rendered video saved at {render_video_save_path}')
# return render_video_save_path


import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
import imageio
from pathlib import Path

from body_visualizer.vis3d.renderer_pytorch3d import Renderer
from body_visualizer.vis3d.smpl import SMPL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input NPZ file path')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--fps', type=float, default=30, help='Output video FPS')
    parser.add_argument('--img-size', type=int, default=800, help='Output image size')
    parser.add_argument('--focal-length', type=float, default=5000, help='Camera focal length')
    parser.add_argument('--smpl-path', type=str, default='data/smpl', help='SMPL model path')
    parser.add_argument('--colors-path', type=str, default='data/colors.txt', help='Colors file path')
    return parser.parse_args()

def process_npz(npz_path, device):
    data = dict(np.load(npz_path))
    
    # Convert poses to correct format
    poses = data["poses"].reshape(-1, 55, 3)
    
    params = {
        'betas': torch.from_numpy(data["betas"]).float().to(device),
        'transl': torch.from_numpy(data["trans"]).float().to(device),
        'orient': torch.from_numpy(poses[:, 0]).float().to(device),
        'body_pose': torch.from_numpy(poses[:, 1:22]).float().to(device),
        'hand_pose': torch.from_numpy(poses[:, -30:]).float().to(device)
    }
    
    return params

def setup_renderer(img_size, focal_length, device, faces, scale=1.0, center_x=0, center_z=0):
    renderer = Renderer(
        img_size, img_size,
        focal_length,
        device,
        faces,
        bin_size=128,
        max_faces_per_bin=20000
    )
    renderer.set_ground(scale, center_x, center_z)
    return renderer

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize SMPL model
    smpl = SMPL(args.smpl_path, pose_type='body26fk', create_transl=False).to(device)
    
    # Load colors
    colors = torch.from_numpy(np.loadtxt(args.colors_path)/255).float().to(device)
    
    # Process NPZ file
    params = process_npz(args.input, device)
    batch_size = params['transl'].shape[0]
    
    # Setup SMPL parameters
    with torch.no_grad():
        output = smpl(
            return_verts=True,
            return_full_pose=True,
            betas=params['betas'][:10].expand(batch_size, -1),
            body_pose=params['body_pose'].reshape(-1, 63),
            transl=params['transl'],
            global_orient=params['orient'],
            left_hand_pose=params['hand_pose'][:, :15],
            right_hand_pose=params['hand_pose'][:, 15:],
            jaw_pose=torch.zeros(batch_size, 3).to(device),
            leye_pose=torch.zeros(batch_size, 3).to(device),
            reye_pose=torch.zeros(batch_size, 3).to(device),
            expression=torch.zeros(batch_size, 50).to(device)
        )
    
    vertices = output.vertices
    vertices = vertices[:, :, [0, 2, 1]]  # xyz -> xzy
    vertices[:, :, 2] *= -1  # inverse y
    
    # Setup renderer
    renderer = setup_renderer(args.img_size, args.focal_length, device, smpl.faces)
    
    # Setup video writer
    writer = imageio.get_writer(args.output, fps=args.fps, mode='I', format='FFMPEG', macro_block_size=1)
    
    # Render frames
    for i in tqdm(range(batch_size)):
        verts = vertices[i:i+1].to(device)
        verts_colors = colors[0:1].expand(verts.shape[0], -1, -1)
        
        # Create default camera parameters
        cam_R = torch.eye(3)[None].to(device)
        cam_T = torch.tensor([[0, 0, -2.5]]).to(device)
        
        cameras, lights = renderer.create_camera_from_cv(cam_R, cam_T)
        frame = renderer.render_with_ground_multiple(
            verts[:, None],
            renderer.faces.squeeze(0),
            verts_colors,
            cameras,
            lights
        )
        
        writer.append_data(frame)
    
    writer.close()
    print(f'[INFO] Rendered video saved at {args.output}')

if __name__ == '__main__':
    main()