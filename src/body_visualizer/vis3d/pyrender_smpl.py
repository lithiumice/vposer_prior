import os
import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import pyrender
import trimesh
import torch
import cv2
from PIL import Image
import tempfile
import shutil
import os.path as osp
from tqdm import trange
from .torch_transform import quat_apply, quat_between_two_vec, quaternion_to_angle_axis, angle_axis_to_quaternion
from .smpl import SMPL
import pytorch3d.transforms as transforms        


class PyrendererSMPL:
    def __init__(self, init_T=6, enable_shadow=False, anti_aliasing=True, use_floor=True,
                 distance=5, elevation=20, azimuth=0, verbose=True):
        self.use_floor = use_floor
        self.verbose = verbose
        self.distance = distance
        self.elevation = elevation
        self.azimuth = azimuth
        self.scene = None
        self.viewer = None
        self.camera = None
        self.light = None
        self.meshes = []
        self.floor_mesh = None
        
        # Animation control
        self.fr = 0
        self.num_fr = 1
        self.T = init_T
        self.paused = False
        self.reverse = False
        self.repeat = False

    def init_camera(self):
        # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        # # Convert spherical coordinates to Cartesian
        # x = self.distance * np.cos(np.radians(self.elevation)) * np.cos(np.radians(self.azimuth))
        # y = self.distance * np.cos(np.radians(self.elevation)) * np.sin(np.radians(self.azimuth))
        # z = self.distance * np.sin(np.radians(self.elevation))



        # # camera_pose = np.eye(4)
        # # camera_pose[:3, 3] = np.array([0, 0, 30.0]) 
                

        # camera_pose = np.eye(4)
        # camera_pose[:3, 3] = np.array([0, 0, 20.0])
        
        
        # rot_mat = transforms.euler_angles_to_matrix(
        #     torch.tensor([[0, 0, -np.pi/2]]), 
        #     convention="XYZ"
        # ).numpy()[0]
        # camera_pose[:3, :3] = rot_mat
                    


            
        # Convert spherical to cartesian coordinates
        distance = self.distance
        elevation = self.elevation
        azimuth = self.azimuth
        
        print(f"[DEBUG] distance: {distance}, elevation: {elevation}, azimuth: {azimuth}")
        
        x = distance * np.cos(elevation) * np.sin(azimuth)
        y = distance * np.cos(elevation) * np.cos(azimuth)
        z = distance * np.sin(elevation)
        
        print(f"[DEBUG] Camera position: {x}, {y},{z}")
        
        # Camera position
        eye = np.array([x, y, z])
        
        # Get camera pose from trimesh look_at
        center = np.zeros(3)  # Looking at origin
        camera_pose = trimesh.scene.cameras.look_at(
            points=eye.reshape((1, 3)),
            fov=60,
            center=center,
            distance=distance,
        )
        
        print(f"[DEBUG] Camera pose: {camera_pose}")

        # Create pyrender camera
        width, height = self.window_size
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        self.camera_node = self.scene.add(camera, pose=camera_pose)
        
        
        
        
    def init_scene(self):
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        
        # Add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.scene.add(light, pose=np.eye(4))
        
        if self.use_floor:
            floor_mesh = trimesh.creation.box([20.0, 20.0, 0.05])
            floor_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.5, 0.5, 0.5, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.5
            )
            floor_mesh = pyrender.Mesh.from_trimesh(floor_mesh, material=floor_material)
            self.floor_node = self.scene.add(floor_mesh, pose=np.eye(4))

class SMPLVisualizer(PyrendererSMPL):
    def __init__(self, generator_func=None, device=torch.device('cpu'), show_smpl=True, 
                 smpl_model_dir=None, show_skeleton=False, sample_visible_alltime=False, **kwargs):
        super().__init__(**kwargs)
        self.show_smpl = show_smpl
        self.show_skeleton = show_skeleton
        self.smpl = SMPL(smpl_model_dir, pose_type='body26fk', create_transl=False).to(device)
        self.device = device
        self.generator_func = generator_func
        self.smpl_motion_generator = None
        self.sample_visible_alltime = sample_visible_alltime
        self.mesh_nodes = []
        
    def update_smpl_seq(self, smpl_seq=None, mode='gt'):
        if smpl_seq is None:
            try:
                smpl_seq = next(self.smpl_motion_generator)
            except:
                self.smpl_motion_generator = self.generator_func()
                smpl_seq = next(self.smpl_motion_generator)
                
        def totype(x): return x.to(self.device).to(torch.float32)
        smpl_seq = {k: totype(v) for k, v in smpl_seq.items()}
        self.smpl_seq = smpl_seq
        
        key = '' if mode == 'gt' else ('infer_out_' if mode == 'sample' else 'recon_out_')
        
        if f'{key}pose' in smpl_seq:
            pose = smpl_seq[f'{key}pose']
            if mode == 'sample':
                pose = pose.squeeze(0)
                
            if f'{key}trans' in smpl_seq:
                trans = smpl_seq[f'{key}trans']
            else:
                trans = smpl_seq['trans'].repeat((pose.shape[0], 1, 1))
            if mode == 'sample':
                trans = trans.squeeze(0)
                
            shape = smpl_seq['shape'].repeat((pose.shape[0], 1, 1))
            
            self.smpl_motion = self.smpl(
                global_orient=pose[..., :3].view(-1, 3),
                body_pose=pose[..., 3:].view(-1, 69),
                root_trans=trans.view(-1, 3),
                betas=shape.view(-1, 10),
                return_full_pose=True,
                orig_joints=True
            )
            
            self.smpl_verts = self.smpl_motion.vertices.reshape(*pose.shape[:-1], -1, 3)
            self.smpl_joints = self.smpl_motion.joints.reshape(*pose.shape[:-1], -1, 3)
            
        self.fr = 0
        self.num_fr = self.smpl_joints.shape[1]
        self.mode = mode
        self.vis_mask = np.ones(self.num_fr)
        
    def update_scene(self):
        if not hasattr(self, 'scene') or not self.scene:
            self.init_scene()
            self.init_camera()
            
        # Remove existing mesh nodes
        for node in self.mesh_nodes:
            self.scene.remove_node(node)
        self.mesh_nodes.clear()
            
        visible = self.vis_mask[self.fr] == 1.0
        
        if self.show_smpl and hasattr(self, 'smpl_verts'):
            for i in range(self.smpl_verts.shape[0]):
                if visible or i == 0 or self.sample_visible_alltime:
                    vertices = self.smpl_verts[i, self.fr].cpu().numpy()
                    faces = self.smpl.faces
                    
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.7, 0.3, 0.3, 1.0],
                        metallicFactor=0.0,
                        roughnessFactor=0.5
                    )
                    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
                    node = self.scene.add(mesh)
                    self.mesh_nodes.append(node)

    def save_frame(self, fr, img_path):
        self.fr = fr
        self.update_scene()
        
        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
        color, _ = r.render(self.scene)
        
        img = Image.fromarray(color)
        img.save(img_path)
        r.delete()

    def save_animation_as_video(self, video_path, init_args=None, window_size=(800, 800), 
                              enable_shadow=None, fps=30, crf=25, frame_dir=None, cleanup=True):
        self.window_size = window_size
        
        if init_args is not None:
            self.update_smpl_seq(init_args.get('smpl_seq', None), init_args.get('mode', 'gt'))
            
        if frame_dir is None:
            frame_dir = tempfile.mkdtemp(prefix="visualizer3d-")
        else:
            if osp.exists(frame_dir):
                shutil.rmtree(frame_dir)
            os.makedirs(frame_dir)
            
        os.makedirs(osp.dirname(video_path), exist_ok=True)
        
        for fr in trange(self.num_fr):
            if fr >= 5: break
            self.save_frame(fr, f'{frame_dir}/{fr:06d}.jpg')
            
        # Create video from frames
        cmd = f'ffmpeg -y -r {fps} -i {frame_dir}/%06d.jpg -c:v libx264 -preset slow -crf {crf} {video_path}'
        os.system(cmd)
        
        if cleanup:
            shutil.rmtree(frame_dir)