import argparse
import numpy as np
import os
import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import math
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys
sys.path.append(str(Path(__file__).parent.parent)) 
from Gaussians.render import Renderer
from Gaussians.pose_utils import SE3_exp
import torch
from PIL import Image
from utils.utils import RGB2SH, SH2RGB, get_args, create_video_from_images,create_video_from_images_multi_view
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

class DataCollector:
    def __init__(self, task=0, data_augmentation=True, use_gs=False,save_dir='collected_data',
                 case=0, reset_cam=0.01,single_view=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_augmentation = data_augmentation
        self.use_gs = use_gs
        self.root_dir = save_dir
        self.case = case
        self.step = 0
        self.task = task
        self.actions = []
        self.qpos = []
        self.succ = True
        self.reset_cam = reset_cam
        self.single_view = single_view
        gs.init(backend=gs.gpu)
        self.init_3d_scene()

        if self.use_gs:
            if self.single_view:
                self.render = Renderer()
                self.render.update_gaussian_data('exports/single-view-scene/splat-transform.ply')
                self.origin_gaussians = self.render.gaussians
                self.original_sh = self.origin_gaussians.sh.clone()
                self.init_pose()
            else:
                self.render_left = Renderer()
                self.render_left.update_gaussian_data('exports/mult-view-scene/left-transform.ply')
                self.origin_gaussians_left = self.render_left.gaussians
                self.original_sh_left = self.origin_gaussians_left.sh.clone()
  
                self.render_right = Renderer()
                self.render_right.update_gaussian_data('exports/mult-view-scene/right-transform.ply')
                self.origin_gaussians_right = self.render_right.gaussians
                self.original_sh_right = self.origin_gaussians_right.sh.clone()
                self.init_mult_view_pose()
        else:
            self.background_path = "GT.jpg"
            self.bg = cv2.imread(self.background_path)        

        if self.single_view:
            rgb, _, _, _ = self.cam.render(depth=True, rgb=True)
            corners = [rgb[-1, 0],  rgb[-1, -1]]
            self.bg_color = np.mean(corners, axis=0)
        else:
            rgb, _, _, _ = self.cam_left.render(depth=True, rgb=True)
            corners = [rgb[-1, 0],  rgb[-1, -1]]
            self.bg_color = np.mean(corners, axis=0)

    def init_3d_scene(self):
        pos1 = 0.32
        pos2 = 0
        pos3 = 0.1
        self.init_gs()
        self.cube = self.scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(
                file="./assets/objects/tennis_can/tennis_can.obj",
                pos=(pos1, pos2, pos3),
                euler=(-90.0, 0.0, 0.0),
                scale=0.5,
                collision=True,
                visualization=True,
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )
        self.goal = self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.35, 0.0, 0.001),  
                size=(0.1, 0.1, 0.001), 
                collision=True,
                visualization=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(
                color=(0.0, 1.0, 0.0),   
                vis_mode="visual",
            )
        )
        
        self.scene.build()

        self.motors_dof = np.arange(5)        
        self.gripper_dof = np.array([5])      
        self.end_effector = self.arm.get_link("Moving_Jaw")
        
        self.scene.step()
        self.default_pos = self.cube.get_dofs_position().cpu().numpy()

    def load_cam_pose(self):
        if self.single_view:
            self.origin_cam_pos = self.cam.pos
            self.origin_cam_lookat = self.cam.lookat
        else:
            self.origin_cam_pos_left = self.cam_left.pos
            self.origin_cam_lookat_left = self.cam_left.lookat
            self.origin_cam_pos_right = self.cam_right.pos
            self.origin_cam_lookat_right = self.cam_right.lookat

    def init_pose(self):
        self.raster_settings = {
            'image_height': 2542, 'image_width': 1694, 
            'tanfovx': 0.7548988664410912, 'tanfovy': 0.5661741498308184, 
            'bg': torch.tensor([0., 0., 0.], device='cuda:0'), 
            'scale_modifier': 1.0, 
            'viewmatrix': torch.tensor([[ -0.052993107587099075, 0.7775523662567139, -0.6265441179275513,  0.0000],
            [0.9982021450996399, 0.023978855460882187, -0.054695188999176025,  0.0000],
            [ -0.027486294507980347,  -0.628330409526825, -0.7774397730827332,  0.0000],
            [-0.011391866020858288,  -0.2659139931201935,  0.5608986616134644,  1.0000]], device='cuda:0'), 
            'projmatrix': torch.tensor([[ -0.07022272795438766, 1.3733447790145874,  -0.6266067624092102,  -0.6265441179275513],
            [1.3227471113204956, 0.04235243797302246,  -0.05470065772533417,  -0.054695188999176025],
            [ -0.03642290085554123,  -1.1097829341888428,  -0.777517557144165,  -0.7774397730827332],
            [-0.015095697715878487,  -0.4696681797504425, 0.55095374584198, 0.5608986616134644]], device='cuda:0'), 
            'sh_degree': 3, 'campos': torch.tensor([ 0.5576120615005493, 0.048413511365652084,  0.26867949962615967], device='cuda:0'), 'prefiltered': False, 'debug': False,
            'projmatrix_raw': torch.tensor([[1.325129508972168,0.0,0.0,0.0],
                [0.0,1.7662409543991089,0.0,0.0],
                [0.0, 0.0,1.000100016593933,1.0],
                [0.0,0.0,-0.010001000016927719,0.0]], device='cuda:0'), }
        self.origin_raster_settings = self.raster_settings.copy()

    def init_mult_view_pose(self):
        self.raster_settings_left = {
            'image_height': 2542, 'image_width': 1694, 
            'tanfovx': 0.7548988664410912, 'tanfovy': 0.5661741498308184, 
            'bg': torch.tensor([0., 0., 0.], device='cuda:0'), 
            'scale_modifier': 1.0, 
            'viewmatrix': torch.tensor([[ -0.052993107587099075, 0.7775523662567139, -0.6265441179275513,  0.0000],
            [0.9982021450996399, 0.023978855460882187, -0.054695188999176025,  0.0000],
            [ -0.027486294507980347,  -0.628330409526825, -0.7774397730827332,  0.0000],
            [-0.011391866020858288,  -0.2659139931201935,  0.5608986616134644,  1.0000]], device='cuda:0'), 
            'projmatrix': torch.tensor([[ -0.07022272795438766, 1.3733447790145874,  -0.6266067624092102,  -0.6265441179275513],
            [1.3227471113204956, 0.04235243797302246,  -0.05470065772533417,  -0.054695188999176025],
            [ -0.03642290085554123,  -1.1097829341888428,  -0.777517557144165,  -0.7774397730827332],
            [-0.015095697715878487,  -0.4696681797504425, 0.55095374584198, 0.5608986616134644]], device='cuda:0'), 
            'sh_degree': 3, 'campos': torch.tensor([ 0.5576120615005493, 0.048413511365652084,  0.26867949962615967], device='cuda:0'), 'prefiltered': False, 'debug': False,
            'projmatrix_raw': torch.tensor([[1.325129508972168,0.0,0.0,0.0],
                [0.0,1.7662409543991089,0.0,0.0],
                [0.0, 0.0,1.000100016593933,1.0],
                [0.0,0.0,-0.010001000016927719,0.0]], device='cuda:0'), }

        left_first_id = 0
        nerfstudio_dataparser = 'outputs/left-processed/splatfacto/2025-08-26_112829/dataparser_transforms.json'
        import json
        device = self.device
        dtype = torch.float32
        
        with open(nerfstudio_dataparser, 'r') as f:
            nerfstudio_data = json.load(f)
            transform = torch.tensor(nerfstudio_data['transform'], device=device, dtype=dtype)
            scale = torch.tensor(nerfstudio_data['scale'], device=device, dtype=dtype)

        transforms_path = 'images/left-processed/transforms.json'
        with open(transforms_path, 'r') as f:
            transforms_data = json.load(f)
            applied_transform = torch.tensor(transforms_data['applied_transform'], device=device, dtype=dtype)
            applied_transform = torch.cat([applied_transform, torch.tensor([[0,0,0,1]], device=device, dtype=dtype)], 0)
            
            left_transform_nerfstudio = torch.tensor(
                transforms_data['frames'][left_first_id]['transform_matrix'], device=device, dtype=dtype
            )

        transform_o = transform @ torch.inverse(applied_transform)
        left_transform_nerfstudio = transform_o @ left_transform_nerfstudio
        left_transform_nerfstudio[:3, 3] *= scale
        
        left_viewmatrix_nerfstudio = get_viewmat(left_transform_nerfstudio.unsqueeze(0)).T
        left_c2w = left_viewmatrix_nerfstudio.inverse().T

        translation = np.array([-0.34, -0.09, 0.42])  
        translation = np.array([0.34, 0.09, 0.42])

        rotation_degrees = np.array([-34.29, 11.67, -180-47.35]) 
        scale_value = 0.81 
        transform = torch.tensor(
            transform_matrix3(translation, rotation_degrees, scale=scale_value),
            device=left_c2w.device, dtype=left_c2w.dtype
        )
        
        left_c2w_transformed = transform @ left_c2w
        self.left_init_c2w = left_c2w_transformed
        left_viewmatrix_nerfstudio = self.left_init_c2w.inverse().T

        viewmatrix_left = torch.tensor(left_viewmatrix_nerfstudio, device='cuda:0', dtype=torch.float32)
        # viewmatrix_left = torch.tensor(answer_w2c.T, device='cuda:0', dtype=torch.float32)
        cam_pos_left = viewmatrix_left.inverse()[3, :3]
        
        self.raster_settings_left['viewmatrix'] = viewmatrix_left.detach()
        self.raster_settings_left['campos'] = cam_pos_left.detach()
        self.raster_settings_left['projmatrix'] = viewmatrix_left @ self.raster_settings_left['projmatrix_raw']

        self.origin_raster_settings_left = self.raster_settings_left.copy()
        
        # Right camera processing
        right_first_id = 1
        nerfstudio_dataparser = 'outputs/right-processed/splatfacto/2025-08-26_105600/dataparser_transforms.json'
        
        with open(nerfstudio_dataparser, 'r') as f:
            nerfstudio_data = json.load(f)
            transform = torch.tensor(nerfstudio_data['transform'], device=device, dtype=dtype)
            scale = torch.tensor(nerfstudio_data['scale'], device=device, dtype=dtype)

        transforms_path = 'images/right-processed/transforms.json'
        with open(transforms_path, 'r') as f:
            transforms_data = json.load(f)
            applied_transform = torch.tensor(transforms_data['applied_transform'], device=device, dtype=dtype)
            applied_transform = torch.cat([applied_transform, torch.tensor([[0,0,0,1]], device=device, dtype=dtype)], 0)
            
            right_transform_nerfstudio = torch.tensor(
                transforms_data['frames'][right_first_id]['transform_matrix'], device=device, dtype=dtype
            )

        transform_o = transform @ torch.inverse(applied_transform)
        right_transform_nerfstudio = transform_o @ right_transform_nerfstudio
        right_transform_nerfstudio[:3, 3] *= scale
        
        right_viewmatrix_nerfstudio = get_viewmat(right_transform_nerfstudio.unsqueeze(0)).T
        right_c2w = right_viewmatrix_nerfstudio.inverse().T
        
        translation = np.array([-0.363, 0.212, 0.383])  
        translation = np.array([0.363, -0.212, 0.383]) 

        rotation_degrees = np.array([-37.95, 20.8, -180-107.2]) 
        scale_value = 0.721 
        transform = torch.tensor(
            transform_matrix3(translation, rotation_degrees, scale=scale_value),
            device=right_c2w.device, dtype=right_c2w.dtype
        )
        
        right_c2w_transformed = transform @ right_c2w
        self.right_init_c2w = right_c2w_transformed
        right_viewmatrix_nerfstudio = self.right_init_c2w.inverse().T
        
        viewmatrix_right = right_viewmatrix_nerfstudio
        cam_pos_right = viewmatrix_right.inverse()[3, :3]
        
        self.raster_settings_right = {
            'image_height': 2542, 'image_width': 1694, 
            'tanfovx': 0.7548988664410912, 'tanfovy': 0.5661741498308184, 
            'bg': torch.tensor([0., 0., 0.], device='cuda:0'), 
            'scale_modifier': 1.0, 
            'viewmatrix': viewmatrix_right.detach(),
            'campos': cam_pos_right.detach(),
            'sh_degree': 3, 'prefiltered': False, 'debug': False,
            'projmatrix_raw': torch.tensor([[1.325129508972168,0.0,0.0,0.0],
                [0.0,1.7662409543991089,0.0,0.0],
                [0.0, 0.0,1.000100016593933,1.0],
                [0.0,0.0,-0.010001000016927719,0.0]], device='cuda:0'), }
        self.raster_settings_right['projmatrix'] = viewmatrix_right @ self.raster_settings_right['projmatrix_raw']
        self.origin_raster_settings_right = self.raster_settings_right.copy()

    def move_action(self, n_steps, qpos, gripper, mid_noise=0.1, noise=0.05):
        device = qpos.device
        start_q = self.arm.get_dofs_position()[:5].clone()
        mid_point = 0.5 * (start_q + qpos)
        mid_point += torch.randn(5, device=device) * mid_noise
        for i in range(n_steps):
            t = i / n_steps

            if t < 0.5:
                alpha = 2 * t
                alpha = alpha * alpha
                q_interp = (1-alpha)*start_q + alpha*mid_point
            else:
                alpha = 2 * (t - 0.5)
                alpha = 1 - (1 - alpha) * (1 - alpha)
                q_interp = (1-alpha)*mid_point + alpha*qpos

            if t > 0.75:
                perturbation = torch.randn(5, device=device) * 0
            else:         
                max_perturb = min(t, 1-t) * noise
                perturbation = torch.randn(5, device=device) * max_perturb
            q_interp += perturbation
            self.arm.control_dofs_position(q_interp, self.motors_dof)
            self.arm.control_dofs_position([gripper], self.gripper_dof)
            self.scene.step()
            self.get_obs_img(bg_color=self.bg_color, actions=np.append(q_interp.cpu().numpy(), gripper))
            self.step+=1

    def reset(self):
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])

        r_min = 0.25
        r_max = 0.42
        
        theta = random.uniform(-math.pi/2, math.pi/2) 
        r = math.sqrt(random.uniform(r_min**2, r_max**2))  
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        pos = np.r_[x, y, self.default_pos[2:6]]
        self.cube.set_dofs_position(pos)
        self.step = 0
        self.scene.step()
        self.step += 1
        self.case += 1

    def close(self, threshold=0.00001):
        target_grip_pos = -0.1 
        tolerance = 0.005
        max_steps = 500
        for i in range(max_steps):
            self.arm.control_dofs_position(np.array([target_grip_pos]), self.gripper_dof)
            self.scene.step()
            self.get_obs_img(bg_color=self.bg_color, actions=np.append(self.arm.get_dofs_position()[:5].cpu().numpy(), target_grip_pos))
            self.step += 1
            
            current_pos = self.arm.get_dofs_position(self.gripper_dof).cpu().numpy()[0]
            
            if np.abs(current_pos - target_grip_pos) < tolerance:
                break
            
            if i > 10 and np.abs(current_pos - prev_pos) < threshold:
                break
            
            prev_pos = current_pos

    def open(self,target_grip_pos=0.7):
        tolerance = 0.05
        max_steps = 150
        for i in range(max_steps):
            self.arm.control_dofs_position(np.array([target_grip_pos]), self.gripper_dof)
            self.scene.step()

            self.get_obs_img(bg_color=self.bg_color, actions=np.append(self.arm.get_dofs_position()[:5].cpu().numpy(), target_grip_pos))
            self.step += 1

            current_pos = self.arm.get_dofs_position(self.gripper_dof).cpu().numpy()[0]
            
            if np.abs(current_pos - target_grip_pos) < tolerance:
                break

    def random(self):
        # self.random_camera_pose()
        # self.random_gaussians(scale_range_min=0.8, scale_range_max=1.2, offset_range_min=-0.01, offset_range_max=0.01,noise_std=0.005)
        self.random_gaussians(scale_range_min=0.95, scale_range_max=1.05, offset_range_min=-0.005, offset_range_max=0.005,noise_std=0.001)


    def random_camera_pose(self,rot_range=0.02,trans_range=0.02,use_gaussian=False):
        if use_gaussian:
            theta = torch.randn((1, 3), device=self.device) * (rot_range / 3)
            rho = torch.randn((1,3), device=self.device) * (trans_range / 3)
            theta = torch.clamp(theta, -rot_range, rot_range)
            rho = torch.clamp(rho, -trans_range, trans_range)
        else:
            # theta = (torch.rand(3, device=self.device) - 0.5) * 2 * rot_range
            # rho = (torch.rand(3, device=self.device) - 0.5) * 2 * trans_range
            theta = (torch.rand((1, 3), device=self.device).float() - 0.5) * rot_range
            rho = (torch.rand((1, 3), device=self.device).float() - 0.5) * trans_range
        tau = torch.cat([rho, theta], axis=0)


        T_w2c = self.origin_raster_settings['viewmatrix'].T
        assert T_w2c.shape == (4, 4), "View matrix must be 4x4"
        assert torch.allclose(T_w2c[3, :3], torch.zeros(3, device=self.device)), "Last row must be [0,0,0,1]"
        
        delta_T = SE3_exp(tau) 
        new_w2c = delta_T @ T_w2c
        
        self.raster_settings.update({
            'viewmatrix': new_w2c.T,
            'projmatrix': new_w2c.T @ self.origin_raster_settings['projmatrix_raw'],
            'campos': new_w2c.inverse()[:3, 3]  
        })

    def random_gaussians(self,scale_range_min=0.95, scale_range_max=1.05, offset_range_min=-0.005, offset_range_max=0.005,noise_std=0.001):
        if self.single_view:
            original_sh = self.original_sh
            rgb = SH2RGB(original_sh)

            scale = torch.rand(3, device=original_sh.device) * (scale_range_max - scale_range_min) + scale_range_min
            offset = torch.rand(3, device=original_sh.device) * (offset_range_max - offset_range_min) + offset_range_min
            noise = torch.randn_like(rgb) * noise_std

            perturbed_rgb = torch.clamp(
                scale.view(1, 3) * rgb + offset.view(1, 3) + noise,
                0.0, 1.0
            )
            
            self.render.gaussians.sh = RGB2SH(perturbed_rgb)
        else:
            original_sh_left = self.original_sh_left
            rgb_left = SH2RGB(original_sh_left)

            scale = torch.rand(3, device=original_sh_left.device) * (scale_range_max - scale_range_min) + scale_range_min
            offset = torch.rand(3, device=original_sh_left.device) * (offset_range_max - offset_range_min) + offset_range_min
            noise = torch.randn_like(rgb_left) * noise_std

            perturbed_rgb_left = torch.clamp(
                scale.view(1, 3) * rgb_left + offset.view(1, 3) + noise,
                0.0, 1.0
            )

            self.render_left.gaussians.sh = RGB2SH(perturbed_rgb_left)

            original_sh_right = self.original_sh_right
            rgb_right = SH2RGB(original_sh_right)

            scale = torch.rand(3, device=original_sh_right.device) * (scale_range_max - scale_range_min) + scale_range_min
            offset = torch.rand(3, device=original_sh_right.device) * (offset_range_max - offset_range_min) + offset_range_min
            noise = torch.randn_like(rgb_right) * noise_std

            perturbed_rgb_right = torch.clamp(
                scale.view(1, 3) * rgb_right + offset.view(1, 3) + noise,
                0.0, 1.0
            )

            self.render_right.gaussians.sh = RGB2SH(perturbed_rgb_right)

    def get_bg(self):
        if self.single_view:
            genesis_c2w = torch.tensor(self.cam.transform, device='cuda:0', dtype=torch.float32)
            genesis_c2w = gl_to_cv(genesis_c2w)
            
            cam_pos = genesis_c2w[:3, 3]
            viewmatrix = torch.linalg.inv(genesis_c2w).T
            self.raster_settings['viewmatrix'] = viewmatrix.detach()
            self.raster_settings['campos'] = cam_pos.detach()
            
            self.raster_settings['projmatrix'] = viewmatrix @ self.raster_settings['projmatrix_raw']
            
            if not hasattr(self, '_zero_tensor'):
                self._zero_tensor = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

            rendered_image = self.render.draw(self.raster_settings, self._zero_tensor, self._zero_tensor)
            if rendered_image.dim() != 3 or rendered_image.size(0) != 3:
                raise ValueError(f"Expected tensor of shape (3, H, W), got {rendered_image.shape}")
            if rendered_image.is_cuda:
                rendered_image = rendered_image.cpu()

            if rendered_image.dtype == torch.float32 or rendered_image.dtype == torch.float16:
                rendered_image = rendered_image.mul(255).clamp(0, 255).byte()
            elif rendered_image.dtype == torch.uint8:
                pass
            else:
                raise ValueError(f"Unsupported tensor dtype: {rendered_image.dtype}")

            image_array = rendered_image.permute(1, 2, 0).numpy()
            if self.data_augmentation and self.use_gs:
                self.random()
            return image_array
        else:
            # print('raster_settings_left before get bg:\n', self.raster_settings_left)
            genesis_c2w_left = torch.tensor(self.cam_left.transform, device='cuda:0', dtype=torch.float32)
            # print('Left Camera Transform in genesis:\n', genesis_c2w_left.cpu().numpy())
            # print('Left Camera w2c in genesis:\n', np.linalg.inv(genesis_c2w_left.cpu().numpy()))
            genesis_c2w_left = gl_to_cv(genesis_c2w_left)
            # print('Left Camera Transform in opencv:\n', genesis_c2w_left.cpu().numpy())
            # print('Left Camera w2c in opencv:\n', np.linalg.inv(genesis_c2w_left.cpu().numpy()))
            cam_pos_left = genesis_c2w_left[:3, 3]
            viewmatrix_left = torch.linalg.inv(genesis_c2w_left).T
            # print('viewmatrix_left:\n', viewmatrix_left.cpu().numpy())
            self.raster_settings_left['viewmatrix'] = viewmatrix_left.detach()
            self.raster_settings_left['campos'] = cam_pos_left.detach()

            self.raster_settings_left['projmatrix'] = viewmatrix_left @ self.raster_settings_left['projmatrix_raw']
            # print('raster_settings_left after get bg:\n', self.raster_settings_left)

            if not hasattr(self, '_zero_tensor'):
                self._zero_tensor = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

            rendered_image_left = self.render_left.draw(self.raster_settings_left, self._zero_tensor, self._zero_tensor)
            if rendered_image_left.dim() != 3 or rendered_image_left.size(0) != 3:
                raise ValueError(f"Expected tensor of shape (3, H, W), got {rendered_image_left.shape}")
            if rendered_image_left.is_cuda:
                rendered_image_left = rendered_image_left.cpu()
            if rendered_image_left.dtype == torch.float32 or rendered_image_left.dtype == torch.float16:
                rendered_image_left = rendered_image_left.mul(255).clamp(0, 255).byte()
            elif rendered_image_left.dtype == torch.uint8:
                pass
            else:
                raise ValueError(f"Unsupported tensor dtype: {rendered_image_left.dtype}")

            image_array_left = rendered_image_left.permute(1, 2, 0).numpy()

            genesis_c2w_right = torch.tensor(self.cam_right.transform, device='cuda:0', dtype=torch.float32)
            genesis_c2w_right = gl_to_cv(genesis_c2w_right)

            cam_pos_right = genesis_c2w_right[:3, 3]
            viewmatrix_right = torch.linalg.inv(genesis_c2w_right).T
            self.raster_settings_right['viewmatrix'] = viewmatrix_right.detach()
            self.raster_settings_right['campos'] = cam_pos_right.detach()

            self.raster_settings_right['projmatrix'] = viewmatrix_right @ self.raster_settings_right['projmatrix_raw']

            if not hasattr(self, '_zero_tensor'):
                self._zero_tensor = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

            rendered_image_right = self.render_right.draw(self.raster_settings_right, self._zero_tensor, self._zero_tensor)
            if rendered_image_right.dim() != 3 or rendered_image_right.size(0) != 3:
                raise ValueError(f"Expected tensor of shape (3, H, W), got {rendered_image_right.shape}")
            if rendered_image_right.is_cuda:
                rendered_image_right = rendered_image_right.cpu()
            if rendered_image_right.dtype == torch.float32 or rendered_image_right.dtype == torch.float16:
                rendered_image_right = rendered_image_right.mul(255).clamp(0, 255).byte()
            elif rendered_image_right.dtype == torch.uint8:
                pass
            else:
                raise ValueError(f"Unsupported tensor dtype: {rendered_image_right.dtype}")

            image_array_left = rendered_image_left.permute(1, 2, 0).numpy()
            image_array_right = rendered_image_right.permute(1, 2, 0).numpy()

            if self.data_augmentation and self.use_gs:
                self.random()
            return image_array_left, image_array_right

    def get_obs_img(self, tol=10, feather_size=1, bg_color=np.array([32, 32, 186]), actions=None):
        if self.single_view:
            current_transform = self.cam.transform
            if not hasattr(self, '_last_transform'):
                transform_changed = True
                self._last_transform = current_transform
            else:
                transform_changed = not np.array_equal(current_transform, self._last_transform)
                self._last_transform = current_transform

            self.transform = current_transform

            if self.use_gs:
                if transform_changed:
                    bg = self.bg = self.get_bg()
                else:
                    bg = self.bg
            else:
                bg = self.bg

            rgb, _, _, _ = self.cam.render(depth=True, rgb=True)

            if rgb.shape[2] == 3:  
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if rgb.shape[:2] != bg.shape[:2]:
                bg = cv2.resize(bg, (rgb.shape[1], rgb.shape[0]))

            lower = np.array([max(0, c - tol) for c in bg_color], dtype=np.uint8)
            upper = np.array([min(255, c + tol) for c in bg_color], dtype=np.uint8)

            bg_mask = cv2.inRange(rgb, lower, upper)       
            fg_mask = cv2.bitwise_not(bg_mask)

            if feather_size > 1:
                alpha_mask = cv2.GaussianBlur(fg_mask, (feather_size, feather_size), 0) / 255.0
                composite = (rgb * alpha_mask[..., None] + 
                            bg * (1 - alpha_mask[..., None])).astype(np.uint8)
            else:
                composite = bg.copy()
                composite[fg_mask.astype(bool)] = rgb[fg_mask.astype(bool)]

            cv2.imwrite(str(self.save_dir / f'frame_{self.step}.png'), composite)
        else:
            current_transform_left = self.cam_left.transform
            current_transform_right = self.cam_right.transform
            if not hasattr(self, '_last_transform_left'):
                transform_changed = True
                self._last_transform_left = current_transform_left
            else:
                transform_changed = not np.array_equal(current_transform_left, self._last_transform_left)
                self._last_transform_left = current_transform_left

            self.transform_left = current_transform_left

            if not hasattr(self, '_last_transform_right'):
                self._last_transform_right = current_transform_right
            else:
                transform_changed = not np.array_equal(current_transform_right, self._last_transform_right)
                self._last_transform_right = current_transform_right

            self.transform_left = current_transform_left

            if self.use_gs and transform_changed:
                # print('use gs to get bg')
                self.bg_left, self.bg_right = self.get_bg()

            rgb_left, _, _, _ = self.cam_left.render(depth=True, rgb=True)
            rgb_right, _, _, _ = self.cam_right.render(depth=True, rgb=True)

            if rgb_left.shape[2] == 3:  
                rgb_left = cv2.cvtColor(rgb_left, cv2.COLOR_RGB2BGR)
            if rgb_left.shape[:2] != self.bg_left.shape[:2]:
                self.bg_left = cv2.resize(self.bg_left, (rgb_left.shape[1], rgb_left.shape[0]))

            if rgb_right.shape[2] == 3:  
                rgb_right = cv2.cvtColor(rgb_right, cv2.COLOR_RGB2BGR)
            if rgb_right.shape[:2] != self.bg_right.shape[:2]:
                self.bg_right = cv2.resize(self.bg_right, (rgb_right.shape[1], rgb_right.shape[0]))

            lower = np.array([max(0, c - tol) for c in bg_color], dtype=np.uint8)
            upper = np.array([min(255, c + tol) for c in bg_color], dtype=np.uint8)

            bg_mask_left = cv2.inRange(rgb_left, lower, upper)       
            fg_mask_left = cv2.bitwise_not(bg_mask_left)
            bg_mask_right = cv2.inRange(rgb_right, lower, upper)       
            fg_mask_right = cv2.bitwise_not(bg_mask_right)

            if feather_size > 1:
                alpha_mask_left = cv2.GaussianBlur(fg_mask_left, (feather_size, feather_size), 0) / 255.0
                composite_left = (rgb_left * alpha_mask_left[..., None] + 
                            self.bg_left * (1 - alpha_mask_left[..., None])).astype(np.uint8)

                alpha_mask_right = cv2.GaussianBlur(fg_mask_right, (feather_size, feather_size), 0) / 255.0
                composite_right = (rgb_right * alpha_mask_right[..., None] + 
                            self.bg_right * (1 - alpha_mask_right[..., None])).astype(np.uint8)
            else:
                composite_left = self.bg_left.copy()
                composite_left[fg_mask_left.astype(bool)] = rgb_left[fg_mask_left.astype(bool)]
                composite_right = self.bg_right.copy()
                composite_right[fg_mask_right.astype(bool)] = rgb_right[fg_mask_right.astype(bool)]
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_bg_left.png'), self.bg_left)
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_bg_right.png'), self.bg_right)
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_rgb_left.png'), rgb_left)
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_rgb_right.png'), rgb_right)
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_left.png'), composite_left)
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_right.png'), composite_right)
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_left_bg.png'), self.bg_left)
            cv2.imwrite(str(self.save_dir / f'frame_{self.step}_left_sim.png'), rgb_left)
            # print('save frame:', str(self.save_dir / f'frame_{self.step}_left.png'), str(self.save_dir / f'frame_{self.step}_right.png'))

        qpos = self.arm.get_dofs_position().cpu().numpy()

        if actions is None:
            actions = qpos.copy()
        else:
            actions = np.asarray(actions)

        if actions.shape == (5,):
            grip_cur = qpos[5]
            actions = np.append(actions, grip_cur)

        assert actions.shape == (6,), "Actions must be a 6D vector (5 joints + gripper)"
        assert qpos.shape == (6,), "Qpos must be 6D (5 joints + gripper)"

        def turn_into_real_actions(actions):
            actions = actions * 180.0 / np.pi
            actions[0] = -actions[0]
            actions[1] = -actions[1]
            actions[4] = -actions[4]
            return actions

        self.actions.append(turn_into_real_actions(actions))
        self.qpos.append(turn_into_real_actions(qpos))

    def get_pos(self, obj_pos, distance=0.03):
        import math
        x = obj_pos[0]
        y = obj_pos[1]
        d = math.sqrt(x**2 + y**2)
        nx = x / d
        ny = y / d 
        x = x - distance * nx
        y = y - distance * ny    
        return x, y

    def get_data(self, n_steps=50):
        obj_pos = self.cube.get_dofs_position()[:3].cpu().numpy()
        goal_pos = self.goal.get_dofs_position()[:3].cpu().numpy()


        x, y = self.get_pos(obj_pos, distance=0.032)

        q_pos1 = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x, y, obj_pos[2]+0.1]),
            quat=self.get_quat(np.array([x, y, obj_pos[2]])),
        )[:5]

        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q_interp = (1 - alpha) * self.arm.get_dofs_position()[:5].clone() + alpha * q_pos1
            self.arm.control_dofs_position(q_interp, self.motors_dof)
            self.arm.control_dofs_position([2.5], self.gripper_dof)
            self.scene.step()
            self.get_obs_img(actions=np.append(q_interp.cpu().numpy(), 2.5))

            self.step+=1
            

        q_pos2= self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x, y, obj_pos[2]-0.03]),
            quat=self.get_quat(np.array([x, y, obj_pos[2]])),
        )[:5]
     
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q_interp = (1 - alpha) * self.arm.get_dofs_position()[:5].clone() + alpha * q_pos2
            self.arm.control_dofs_position(q_interp, self.motors_dof)
            self.arm.control_dofs_position([2.5], self.gripper_dof)
            self.scene.step()
            self.get_obs_img(actions=np.append(q_interp.cpu().numpy(), 2.5))
            self.step+=1

        self.close()

        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q_interp = (1 - alpha) * self.arm.get_dofs_position()[:5].clone() + alpha * q_pos1
            self.arm.control_dofs_position(q_interp, self.motors_dof)
            self.arm.control_dofs_position([2.5], self.gripper_dof)
            self.arm.control_dofs_position(np.array([-0.5]), self.gripper_dof)
            self.scene.step()
            self.get_obs_img(actions=np.append(q_interp.cpu().numpy(), -0.5))
            self.step += 1


        q_pos3= self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([goal_pos[0], goal_pos[1], obj_pos[2]+0.05]),
            quat=self.get_quat(np.array([goal_pos[0], goal_pos[1], obj_pos[2]])),
        )[:5]
    
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q_interp = (1 - alpha) * self.arm.get_dofs_position()[:5].clone() + alpha * q_pos3
            self.arm.control_dofs_position(q_interp, self.motors_dof)
            self.arm.control_dofs_position([2.5], self.gripper_dof)
            self.arm.control_dofs_position(np.array([-0.5]), self.gripper_dof)
            self.scene.step()
            self.get_obs_img(actions=np.append(q_interp.cpu().numpy(), -0.5))
            self.step+=1


        q_pos4= self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([goal_pos[0], goal_pos[1], obj_pos[2]-0.03]),
            quat=self.get_quat(np.array([goal_pos[0], goal_pos[1], obj_pos[2]])),
        )[:5]
    
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q_interp = (1 - alpha) * self.arm.get_dofs_position()[:5].clone() + alpha * q_pos4
            self.arm.control_dofs_position(q_interp, self.motors_dof)
            self.arm.control_dofs_position([2.5], self.gripper_dof)
            self.arm.control_dofs_position(np.array([-0.5]), self.gripper_dof)
            self.scene.step()
            self.get_obs_img(actions=np.append(q_interp.cpu().numpy(), -0.5))
            self.step+=1

        self.open()

    def reset_scene(self, reset_cam=0.01):
        if self.single_view:
            pos_noise = np.random.uniform(-reset_cam, reset_cam, size=3)
            lookat_noise = np.random.uniform(-reset_cam, reset_cam, size=3)
            self.cam.set_pose(pos=self.origin_cam_pos + pos_noise, lookat=self.origin_cam_lookat + lookat_noise)
        else:
            pos_noise_left = np.random.uniform(-reset_cam, reset_cam, size=3)
            lookat_noise_left = np.random.uniform(-reset_cam, reset_cam, size=3)
            self.cam_left.set_pose(pos=self.origin_cam_pos_left + pos_noise_left, lookat=self.origin_cam_lookat_left + lookat_noise_left)

            pos_noise_right = np.random.uniform(-reset_cam, reset_cam, size=3)
            lookat_noise_right = np.random.uniform(-reset_cam, reset_cam, size=3)
            self.cam_right.set_pose(pos=self.origin_cam_pos_right + pos_noise_right, lookat=self.origin_cam_lookat_right + lookat_noise_right)

        self.scene.vis_options.lights[0]['color'] = np.random.uniform(0.0, 1.0, size=3).tolist() 
        self.scene.vis_options.lights[0]['intensity'] = float(np.random.uniform(1, 30))  

    def run(self, num_steps=1000):
        step = 0
        success = 0
        all_cases = 0
        # self.reset()

        if self.single_view:
            rgb, _, _, _ = self.cam.render(depth=True, rgb=True)
            corners = [rgb[-1, 0],  rgb[-1, -1]]
            self.bg_color = np.mean(corners, axis=0)
        else:
            rgb, _, _, _ = self.cam_left.render(depth=True, rgb=True)
            corners = [rgb[-1, 0],  rgb[-1, -1]]
            self.bg_color = np.mean(corners, axis=0)

        while True:
            self.save_dir = Path(self.root_dir) / str(self.task) / str(self.case)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.get_data()
            actions = np.array(self.actions)
            qpos = np.array(self.qpos)
            assert actions.shape[0] == qpos.shape[0], "Actions and qpos must have the same number of steps"

            np.save(str(self.save_dir / f'actions.npy'), actions)
            np.save(str(self.save_dir / f'qpos.npy'), qpos)
            self.reset()
            if not self.succ:
                all_cases += 1
                self.case -= 1
                self.actions = []
                self.qpos = []
                try:
                    shutil.rmtree(self.save_dir)
                    print(f"remove: {self.save_dir}")
                except FileNotFoundError:
                    print(f"no path: {self.save_dir}")
                continue
            else:
                if self.single_view:
                    create_video_from_images(self.save_dir, self.save_dir/'video.mp4', lossless=False)
                else:
                    create_video_from_images_multi_view(self.save_dir, self.save_dir/'video_left.mp4', lossless=False,view='left')
                    create_video_from_images_multi_view(self.save_dir, self.save_dir/'video_right.mp4', lossless=False,view='right')
                step += 1
                success += 1
                all_cases += 1
                self.actions = []
                self.qpos = []

            if step >= num_steps:
                print(f"Collected {step} episodes for task {self.task}")
                print(f"Success rate: {success / all_cases:.2f}")
                break
            
            self.reset_scene(reset_cam=self.reset_cam)
                
        print(f"Case {self.case} finished")

    def init_gs(self, ):
        self.resolution = (640, 480)
        # self.resolution = (2542, 1694)
        self.scene = gs.Scene(
            show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                # res=self.resolution,
                res=(128,128),
                camera_pos=(0.616, 0, 0.415),
                camera_lookat=(0.316, 0, 0),
                camera_fov=56.617,
                max_FPS=120,
            ),
            sim_options=gs.options.SimOptions(
                # dt=0.01,
                substeps=60,
            ),
            show_viewer=True,
            vis_options=gs.options.VisOptions(
                background_color=(1.0, 0.0, 0.0),
                show_world_frame=True,
                world_frame_size=0.35,
                visualize_mpm_boundary=False,
                plane_reflection=False,
                shadow=False,
                ambient_light=(0.1, 0.1, 0.1),
                # ambient_light=(0.4, 0.3, 0.2),
                # ambient_light=(0.2, 0.2, 0.2),
                lights=[
                    {
                        'type': 'directional',
                        'dir': (-1, -1, -1),
                        'color': (1.0, 1.0, 1.0),
                        'intensity': 3.0
                    }
                ]
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.1, -0.3, -0.1),
                upper_bound=(0.65, 0.3, 0.2),
            ),
            renderer=gs.renderers.Rasterizer(),
        )
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 1.0),
                emissive=(1.0, 1.0, 1.0),
                roughness=1.0,
                metallic=0.0,
                vis_mode='visual'
                ),
        )
        if self.single_view:
            self.cam = self.scene.add_camera(
                res=self.resolution,
                pos=(0.516, 0, 0.315),
                lookat=(0.316, 0, 0),
                # pos=(0.0, 0.5, 0.315),
                # lookat=(0, 0, 0),
                # fov = 56.617,
                fov=58.86,
                aperture=2.8,
                GUI=False,
            )
        else:
            left_first_id = 0
            nerfstudio_dataparser = 'outputs/left-processed/splatfacto/2025-08-26_112829/dataparser_transforms.json'
            import json
            device = self.device
            dtype = torch.float32
            
            with open(nerfstudio_dataparser, 'r') as f:
                nerfstudio_data = json.load(f)
                transform = torch.tensor(nerfstudio_data['transform'], device=device, dtype=dtype)
                scale = torch.tensor(nerfstudio_data['scale'], device=device, dtype=dtype)

            transforms_path = 'images/left-processed/transforms.json'
            with open(transforms_path, 'r') as f:
                transforms_data = json.load(f)
                applied_transform = torch.tensor(transforms_data['applied_transform'], device=device, dtype=dtype)
                applied_transform = torch.cat([applied_transform, torch.tensor([[0,0,0,1]], device=device, dtype=dtype)], 0)
                
                left_transform_nerfstudio = torch.tensor(
                    transforms_data['frames'][left_first_id]['transform_matrix'], device=device, dtype=dtype
                )

            transform_o = transform @ torch.inverse(applied_transform)
            left_transform_nerfstudio = transform_o @ left_transform_nerfstudio
            left_transform_nerfstudio[:3, 3] *= scale
            
            left_viewmatrix_nerfstudio = get_viewmat(left_transform_nerfstudio.unsqueeze(0)).T
            left_c2w = left_viewmatrix_nerfstudio.inverse().T

            translation = np.array([-0.34, -0.09, 0.42]) 
            translation = np.array([0.34, 0.09, 0.42])  

            rotation_degrees = np.array([-34.29, 11.67, -180-47.35])  
            scale_value = 0.81 
            transform = torch.tensor(
                transform_matrix3(translation, rotation_degrees, scale=scale_value),
                device=left_c2w.device, dtype=left_c2w.dtype
            )
        
            left_c2w_transformed = transform @ left_c2w

            self.left_init_c2w = left_c2w_transformed

            right_first_id = 1
            nerfstudio_dataparser = 'outputs/right-processed/splatfacto/2025-08-26_105600/dataparser_transforms.json'
            
            with open(nerfstudio_dataparser, 'r') as f:
                nerfstudio_data = json.load(f)
                transform = torch.tensor(nerfstudio_data['transform'], device=device, dtype=dtype)
                scale = torch.tensor(nerfstudio_data['scale'], device=device, dtype=dtype)

            transforms_path = 'images/right-processed/transforms.json'
            with open(transforms_path, 'r') as f:
                transforms_data = json.load(f)
                applied_transform = torch.tensor(transforms_data['applied_transform'], device=device, dtype=dtype)
                applied_transform = torch.cat([applied_transform, torch.tensor([[0,0,0,1]], device=device, dtype=dtype)], 0)
                
                right_transform_nerfstudio = torch.tensor(
                    transforms_data['frames'][right_first_id]['transform_matrix'], device=device, dtype=dtype
                )

            transform_o = transform @ torch.inverse(applied_transform)
            right_transform_nerfstudio = transform_o @ right_transform_nerfstudio
            right_transform_nerfstudio[:3, 3] *= scale
            
            right_viewmatrix_nerfstudio = get_viewmat(right_transform_nerfstudio.unsqueeze(0)).T
            right_c2w = right_viewmatrix_nerfstudio.inverse().T

            translation = np.array([-0.363, 0.212, 0.383])  # supersplat数值
            translation = np.array([0.363, -0.212, 0.383])  # 实际用

            rotation_degrees = np.array([-37.95, 20.8, -180-107.2])  # supersplat数值 实际用
            scale_value = 0.721  # supersplat数值 实际用
            transform = torch.tensor(
                transform_matrix3(translation, rotation_degrees, scale=scale_value),
                device=right_c2w.device, dtype=right_c2w.dtype
            )
        
            right_c2w_transformed = transform @ right_c2w

            self.right_init_c2w = right_c2w_transformed

            left_transform_colmap = self.left_init_c2w
            right_transform_colmap = self.right_init_c2w

            left_transform_opengl = cv_to_gl(left_transform_colmap)
            right_transform_opengl = cv_to_gl(right_transform_colmap)
            pos_left, lookat_left, right_left, up_left = get_camera_params(left_transform_opengl)
            pos_right, lookat_right, right_right, up_right = get_camera_params(right_transform_opengl)
            # print("Left Camera:")
            # print("Position:", pos_left)
            # print("LookAt:", lookat_left)
            # print("Up:", up_left)
            # print("colmap Transform:\n", left_transform_colmap.numpy())
            # print("Right Camera:")
            # print("Position:", pos_right)
            # print("LookAt:", lookat_right)
            # print("Up:", up_right)
            # print("colmap Transform:\n", right_transform_colmap.numpy())

            self.cam_left = self.scene.add_camera(
                res=self.resolution,
                pos=pos_left.cpu().numpy(),
                lookat=lookat_left.cpu().numpy(),
                up=up_left.cpu().numpy(),
                fov=58.86,
                aperture=2.8,
                GUI=False,
            )

            self.cam_right = self.scene.add_camera(
                res=self.resolution,
                pos=pos_right.cpu().numpy(),
                lookat=lookat_right.cpu().numpy(),
                up=up_right.cpu().numpy(),
                fov=58.86,
                aperture=2.8,
                GUI=False,
            )
            # print("Left Camera Transform in opengl :\n", np.array(self.cam_left.transform))
            # print("Right Camera Transform in opengl:\n", np.array(self.cam_right.transform))

        self.arm = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file="./assets/so100/urdf/so_arm100.xml",
                euler=(0.0, 0.0, 90.0),
                pos=(0.0, 0.0, 0.0),
            ),
            material=gs.materials.Rigid(friction=4),
        )

    def get_quat(self, banana_quat, vertical_rotation=90.0):
        banana_rot = R.from_quat([banana_quat[1], banana_quat[2], banana_quat[3], banana_quat[0]])    
        rot_matrix = banana_rot.as_matrix()
        obj_rz_rad = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
        
        grip_rz_rad = obj_rz_rad + np.radians(vertical_rotation)
        grip_rz_deg = np.degrees(grip_rz_rad)
        # print('grip_rz_deg',grip_rz_deg)
        # # grip_rz_deg = (grip_rz_deg + 90) % 180 - 90  # 规范到[-90,90]
        # print('grip_rz_deg',grip_rz_deg)
        # import pdb; pdb.set_trace()
        # print(f"grip_rz_deg: {grip_rz_deg}")
        # import pdb; pdb.set_trace()
        grip_rz_rad = np.radians(grip_rz_deg)

        base_rot = R.from_euler('y', 90, degrees=True)  # 朝向-Z方向
        grip_rot = R.from_rotvec([0, 0, grip_rz_rad]) * base_rot

        z_axis = grip_rot.apply([0, 0, 1])
        if z_axis[2] > 0:  # 如果意外指向+Z方向
            correction = R.from_euler('x', 180, degrees=True)
            grip_rot = correction * grip_rot

        quat = grip_rot.as_quat()
        return [quat[3], quat[0], quat[1], quat[2]]

    def get_quat_grasp(self,roll_angle_deg: float,forward_axis: str = 'x',quat_format: str = 'xyzw') -> np.ndarray:
        if forward_axis.lower() == 'x':
            base_rotation = Rotation.from_rotvec([0, -np.pi/2, 0])
        elif forward_axis.lower() == 'z':
            base_rotation = Rotation.from_rotvec([0, np.pi, 0])
        else:
            raise ValueError("`forward_axis` 必须是 'x' 或 'z'")

        roll_angle_rad = np.deg2rad(roll_angle_deg)
        roll_rotation = Rotation.from_rotvec([0, 0, -roll_angle_rad])
        final_rotation = roll_rotation * base_rotation

        if quat_format.lower() == 'wxyz':
            q = final_rotation.as_quat()
            return np.array([q[3], q[0], q[1], q[2]]) # [w, x, y, z]
        elif quat_format.lower() == 'xyzw':
            return final_rotation.as_quat() # [x, y, z, w]
        else:
            raise ValueError("`quat_format` must be 'wxyz' or 'xyzw'")

    def get_quat_pick(self, banana_quat):
        banana_rot = R.from_quat([banana_quat[1], banana_quat[2], banana_quat[3], banana_quat[0]])
    
        banana_euler = banana_rot.as_euler('xyz', degrees=True)
        rx, ry, rz = banana_euler
        rz = 180
        if rz > 0:
            rz = rz - 180
        else:
            rz = rz + 180
        new_rot = R.from_euler('xyz', [rz, 90, 90], degrees=True)
        correction_rot = R.from_euler('z', 180, degrees=True)
        new_rot = correction_rot * new_rot
        
        quat = new_rot.as_quat()
        return [quat[3], quat[0], quat[1], quat[2]]

def gl_to_cv(c2w_gl):
    convert_mat = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=c2w_gl.dtype, device=c2w_gl.device)

    c2w_cv = torch.matmul(c2w_gl, convert_mat)
    return c2w_cv

def cv_to_gl(c2w_cv):
    convert_mat = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=c2w_cv.dtype, device=c2w_cv.device)

    c2w_gl = torch.matmul(c2w_cv, convert_mat)
    return c2w_gl

def get_camera_params(c2w_gl):
    # 相机位置
    pos = c2w_gl[:3, 3]
    look_dir = pos-c2w_gl[:3, 2]
    right_dir = c2w_gl[:3, 0]
    up_dir = c2w_gl[:3, 1]
    
    return pos, look_dir, right_dir, up_dir

def build_transform(translation, rotation, degrees=True, s=1.0, order="xyz"):
    tx, ty, tz = translation
    rx, ry, rz = rotation
    r = R.from_euler(order, [rx, ry, rz], degrees=degrees)
    R_mat = r.as_matrix()
    
    T = np.eye(4)
    T[:3,:3] = s * R_mat 
    T[:3,3] = [tx, ty, tz]
    return T

def build_transform2(translation, rotation, degrees=True, s=1.0, order="xyz"):
    tx, ty, tz = translation
    rx, ry, rz = rotation
    r = R.from_euler(order, [rx, ry, rz], degrees=degrees)
    R_mat = r.as_matrix()
    
    T = np.eye(4)
    T[:3,:3] = s * R_mat 
    t = np.array([[tx, ty, tz]]).T
    T[:3,3] = (R_mat@t).T 
    return T

def rotation_matrix(axis, angle_deg):
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
def transform_matrix2(translation, rotation):
    T_translate = np.eye(4)
    T_translate[:3, 3] = translation

    center = translation  

    T_neg = np.eye(4)
    T_neg[:3, 3] = -center

    T_pos = np.eye(4)
    T_pos[:3, 3] = center

    rx, ry, rz = rotation 
    R_x = rotation_matrix('x', rx)
    R_y = rotation_matrix('y', ry)
    R_z = rotation_matrix('z', rz)

    R = T_pos @ R_x @ T_neg
    R = T_pos @ R_y @ T_neg @ R
    R = T_pos @ R_z @ T_neg @ R
    T_final = R @ T_translate

    return T_final

def scale_matrix(s):
    if np.isscalar(s):
        sx = sy = sz = s
    else:
        sx, sy, sz = s
    return np.array([
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1]
    ])

def transform_matrix3(translation=(0.5,0.5,-0.5), 
                     rotation=(30,60,-180), 
                     scale=1.0):
    T_translate = np.eye(4)
    T_translate[:3, 3] = translation

    center = translation  

    T_neg = np.eye(4)
    T_neg[:3, 3] = -center

    T_pos = np.eye(4)
    T_pos[:3, 3] = center

    rx, ry, rz = rotation  
    R_x = rotation_matrix('x', rx)
    R_y = rotation_matrix('y', ry)
    R_z = rotation_matrix('z', rz)

    R = T_pos @ R_x @ T_neg
    R = T_pos @ R_y @ T_neg @ R
    R = T_pos @ R_z @ T_neg @ R

    S = scale_matrix(scale)
    S_center = T_pos @ S @ T_neg

    T_final =S_center@ R @ T_translate

    return T_final

def get_viewmat(optimized_camera_to_world):
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat[0]

if __name__ == "__main__":
    collector = DataCollector(data_augmentation=False, use_gs=True)
    collector.run(num_steps=2)