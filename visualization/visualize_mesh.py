import os
cwd = os.getcwd()
import sys
import argparse
sys.path.append(cwd)
import glob
import random
import pickle
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from smplx import SMPL

import colorsys
import pyrender
import trimesh
from pyrender.constants import RenderFlags
import imageio
import cv2
from scipy.spatial.transform import Rotation as R
import math
from math import factorial

def convert_smpl2joints(
    smpl_model, body_pose, **kwargs
):
    """
    :param smplx_model: 
    :param body_pose: [batch_size, nframes, 75]
    """
    B, T = body_pose.shape[:2]
    device = body_pose.device
    
    transl = body_pose[..., :3]
    global_orient = body_pose[..., 3:6]
    body_pose = body_pose[..., 6:]
    
    output = smpl_model(
        global_orient=global_orient.reshape(B*T, 1, -1), 
        body_pose=body_pose.reshape(B*T, -1, 3), 
        transl=transl.reshape(B*T, -1)
    )
    
    joints = output.joints.reshape(B, T, -1, 3)
    vertices3d = output.vertices.reshape(B, T, -1, 3)
    
    return {"joints": joints[:, :, :24], "vertices": vertices3d}

class MeshRenderer(object):
    def __init__(self, faces, img_h=480, img_w=640, yfov=5.0, x_angle=60):
        self.img_h = img_h
        self.img_w = img_w
        self.faces = faces
        self.cam = pyrender.PerspectiveCamera(yfov=(np.pi / yfov))    
        self.cam_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -4.5],
            [0.0, 0.0, 1.0, 3.75],
            [0.0, 0.0, 0.0, 1.0]
        ])
        Rot = R.from_euler("X", angles=x_angle, degrees=True).as_matrix()
        self.cam_pose[:3, :3] = Rot
        
        self.direc_l = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=5.0)
        
        self.spot_l = pyrender.SpotLight(color=[.5, .1, .1], intensity=10.0,
                           innerConeAngle=np.pi / 16, outerConeAngle=np.pi /12)
        
        self.scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4, 1.0]))
        self.scene.add(self.direc_l)
        self.scene.add(self.spot_l)
        self.scene.add(self.cam, pose=self.cam_pose)
    
    @staticmethod
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number") 
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')
    
    @staticmethod
    def put_to_origin(vertices):
        offset_x = vertices[0, 0, 0]
        offset_y = vertices[0, 0, 1]
        offset_z = np.min(vertices[0, :, 2])
        vertices[..., 0] -= offset_x
        vertices[..., 1] -= offset_y
        vertices[..., 2] -= offset_z
        return vertices
    
    @staticmethod
    def get_offsets(vertices):
        offset_x = vertices[:, 0, 0] - vertices[:1, 0, 0]
        offset_y = vertices[:, 0, 1] - vertices[:1, 0, 1]
        offset_z = np.zeros_like(offset_x)
        return np.stack([offset_x, offset_y, offset_z], axis=-1)
    
    def render(self, vertices, output_path, faces=None):
        """
        :param joints: [N, J, 3]
        :param vertices: [N, V, 3]
        """
        temp_output_path = os.path.join(output_path, "temp")
        if not os.path.exists(temp_output_path):
            os.makedirs(temp_output_path)
                        
        # Put to origin
        # vertices = self.put_to_origin(vertices)
        offsets = self.get_offsets(vertices)
            
        r = pyrender.OffscreenRenderer(viewport_width=self.img_w * 2, 
                                       viewport_height=self.img_h * 2)
                
        plane = trimesh.creation.box(extents=(4., 4., 0.01))
        # plane_color = np.array([[44/255,44/255,44/255]]).repeat(plane.faces.shape[0], axis=0)
        # plane.visual.face_colors = plane_color
        plane = pyrender.Mesh.from_trimesh(plane, smooth=False)
        self.scene.add(plane)
        
        for idx, (verts, offset) in tqdm(enumerate(zip(vertices, offsets)), desc="Rendering..."):
            if faces is None:
                mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
            else:
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            # mesh_face_color = np.array([[229/255, 204/255, 255/255]]).repeat(mesh.faces.shape[0], axis=0)
            mesh_face_color = np.array([[0, 0.706, 1]]).repeat(mesh.faces.shape[0], axis=0)
            mesh.visual.face_colors = mesh_face_color
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            
            mesh_node = pyrender.Node(mesh=mesh, translation=np.array([-offset[0], -offset[1], -offset[2]]))
            
            self.scene.add_node(mesh_node)
            
            flags = RenderFlags.SHADOWS_DIRECTIONAL    
            color, depth = r.render(self.scene, flags=flags)
            cv2.imwrite(os.path.join(temp_output_path, "pred_{:d}.png".format(idx)), color)
            self.scene.remove_node(mesh_node)
            
    def animate(self, output_path, output_name, fps=30):
        
        temp_output_path = os.path.join(output_path, "temp")
        
        files = [int(f.split(".")[0].split("_")[-1]) for f in os.listdir(temp_output_path) if ".png" in f]
        files = sorted(list(set(files)))
        
        with imageio.get_writer(os.path.join(output_path, output_name+".mp4"), fps=fps) as writer:
            for file in tqdm(files, desc="Generating video"):
                img = np.array(imageio.imread(os.path.join(temp_output_path, "pred_{:d}.png".format(file))), dtype=np.uint8)
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                os.remove(os.path.join(temp_output_path, "pred_{:d}.png".format(file)))
                writer.append_data(img[..., :3])
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='results/eval/ude/output/a2m', help='path of generated data')
    parser.add_argument('--output_folder', type=str, default='results/eval/ude/animation/a2m', help='path of generated videos')
    parser.add_argument('--fps', type=int, default=20, help='FPS of rendered videos')
    args = parser.parse_args()
    return args

def main(args):
    
    SMPLModel = SMPL(model_path="./networks/smpl", gender="NEUTRAL", batch_size=1)
    faces = SMPLModel.faces
    renderer = MeshRenderer(faces=faces, img_h=1024, img_w=1024, yfov=5.0, x_angle=60)
    
    files = [f for f in os.listdir(args.input_folder) if ".npy" in f]
    for file in files:
        data = np.load(os.path.join(args.input_folder, file), allow_pickle=True).item()
        smpl_pose = torch.from_numpy(data.get("motion")).float()
        smpl_output = convert_smpl2joints(SMPLModel, body_pose=smpl_pose)
        smpl_verts = smpl_output.get("vertices", None)
        renderer.render(vertices=smpl_verts[0].data.cpu().numpy(), output_path=args.output_folder)
        renderer.animate(output_path=args.output_folder, output_name=file.replace(".npy", ".mp4"), fps=args.fps)
        
if __name__ == "__main__":
    args = parse_args()
    main(args=args)
        