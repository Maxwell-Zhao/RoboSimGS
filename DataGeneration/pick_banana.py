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
from utils.utils import RGB2SH, SH2RGB, get_args
from tasks.base_task import DataCollector
import shutil
import json

class PickBanana(DataCollector):
    def __init__(self, task='pick_banana1', data_augmentation=False, use_gs=True, save_dir='collected_data',
                 case=0,reset_cam=0.01,single_view=False):
        super().__init__(task=task, data_augmentation=data_augmentation, use_gs=use_gs, save_dir=save_dir,
                         case=case, reset_cam=reset_cam,single_view=single_view)

        self.reset()


    def init_3d_scene(self):
        self.init_gs()

        self.banana = self.scene.add_entity(
            # material=gs.materials.MPM.Elastic(E=1e5, rho=100),
            material=gs.materials.Rigid(friction=5),
            morph=gs.morphs.Mesh(
                # file="./assets/objects/toy/toy.obj",
                file="./assets/objects/banana/banana.obj",
                pos=(0.32, 0.1, 0.04),
                euler=(0.0, 0.0, 250.0),
                scale=0.95,
                # collision=True,
                # visualization=True,
                convexify=False,
                decimate=False,
                decompose_nonconvex=True,
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )

        self.box = self.scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(
                file="./assets/objects/box/box.obj",
                pos=(0.2, -0.15, -0.003),
                euler=(90.0, 0.0, 180),
                scale=0.95,
                collision=True,
                visualization=True,
                fixed=True,
                decompose_nonconvex=True,
                decimate=False,
                convexify=False,
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )
        self.box_x = 0.2
        self.box_y = -0.15

        self.scene.build()
        # print("Left Camera Transform in opengl :\n", np.array(self.cam_left.transform))
        # print("Right Camera Transform in opengl:\n", np.array(self.cam_right.transform))
        self.origin_euler = self.banana.get_quat()
        # self.arm.set_dofs_position([0, -1.57, 1.57, 1.57, -1.57, 0])
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        # self.arm.set_dofs_position([0,-3.35, 1.83, 0, 0, 2.5])
        self.scene.step()

        # self.arm.set_dofs_kp(np.array([150, 150, 120, 100, 80, 20]))  
        # self.arm.set_dofs_kv(np.array([50, 50, 45, 40, 30, 8]))     
        # self.arm.set_dofs_force_range(
        #     np.array([-10, -10, -8, -5, -5, -2]),                     
        #     np.array([10, 10, 8, 5, 5, 2])                              
        # )
        self.arm.set_dofs_kp(np.array([300, 300, 250, 200, 160, 40]))  
        self.arm.set_dofs_kv(np.array([30, 30, 25, 20, 15, 10]))     
        self.arm.set_dofs_force_range(
            np.array([-20, -20, -15, -10, -10, -2]),                     
            np.array([20, 20, 15, 10, 10, 7])                              
        )

        self.motors_dof = np.arange(5)
        self.gripper_dof = np.array([5])
        self.end_effector = self.arm.get_link("Moving_Jaw")
        self.default_pos = self.banana.get_dofs_position().cpu().numpy()
        self.load_cam_pose()
        
    def get_data(self, n_steps=37):
        # stay a while 
        for i in range(50):
            self.arm.set_dofs_position([0, -3.32, 3.11, 1.18, 0, -0.174])
            self.scene.step()

        ###################################################
        obj_pos = self.banana.get_pos().cpu().numpy()[:3]
        x, y = self.get_pos(obj_pos, distance=0.01)
        banana_quat = self.banana.get_quat().cpu().numpy()
        banana_rot = R.from_quat([banana_quat[1], banana_quat[2], banana_quat[3], banana_quat[0]])
        banana_euler = banana_rot.as_euler('xyz', degrees=True)
        rx, ry, rz = banana_euler
        rz = rz % 360
        rz = rz % 180

        # print('rz',rz)
        # if rz < 250:
        #     rz = rz - 90
        # else:
        #     rz = rz -180

        rot_grasp = self.get_quat_grasp(roll_angle_deg=rz)[:5]
        q_pos_grasp = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x, y, 0.086]),
            quat=rot_grasp,
            rot_tol=0.08,
            pos_tol=0.02,
            max_solver_iters=500,
        )[:5]

        q_pos_reach = q_pos_grasp.clone()
        q_pos_reach[1] = q_pos_reach[1] - 0.8
        q_pos_reach[2] = q_pos_reach[2] - 0.2
        q_pos_reach[3] = q_pos_reach[3] + 0.3

        ###################################################
        self.move_action(n_steps, q_pos_reach, 0.9)
        self.move_action(n_steps, q_pos_grasp, 0.9, mid_noise=0.01, noise=0.01)

        self.close(0.0001)

        self.move_action(n_steps, q_pos_reach, 0)


        ###################################################
        q_pos_place= self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([self.box_x, self.box_y, 0.05]),
            quat=rot_grasp,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=700,
        )[:5]

        q_pos_place[1] = q_pos_place[1] - 0.5
        q_pos_place[2] = q_pos_place[2] + 0.2
        q_pos_place_reach = q_pos_place.clone()
        q_pos_place_reach[1] = q_pos_place_reach[1] - 0.5
        q_pos_place_reach[2] = q_pos_place_reach[2] + 0.2

        self.move_action(n_steps, q_pos_place_reach, 0)
        self.move_action(n_steps, q_pos_place, 0)

        self.open(0.8)

        self.move_action(n_steps, q_pos_place_reach, 0.8)
        
        final_pose = torch.tensor([0, -3.32, 3.11, 1.18, 0, -0.174], device=self.device)
        self.move_action(n_steps, final_pose[:5], -0.174)

    
        if math.sqrt((self.banana.get_pos().cpu().numpy()[:3][0] - self.box_x)**2 + (self.banana.get_pos().cpu().numpy()[:3][1] - self.box_y)**2) < 0.08:
            self.succ = True
        else:
            self.succ = False

    def reset(self):
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])

        r_min = 0.25
        r_max = 0.35

        theta_banana = random.uniform(1*math.pi/12, 1*math.pi/5)
        r_banana = math.sqrt(random.uniform(r_min**2, r_max**2))
        x_banana = r_banana * math.cos(theta_banana)
        y_banana = r_banana * math.sin(theta_banana)

        r_min = 0.28
        r_max = 0.35
        theta_box = random.uniform(-1*math.pi/5, -1*math.pi/8) 
        r_box = math.sqrt(random.uniform(r_min**2, r_max**2))  
        x_box = r_box * math.cos(theta_box)
        y_box = r_box * math.sin(theta_box)

        pos_banana = np.r_[x_banana, y_banana, 0.04]
        pos_box = np.r_[x_box, y_box, -0.004]

        self.box_x = x_box
        self.box_y = y_box
        self.banana_x = x_banana
        self.banana_y = y_banana

        if math.sqrt((self.banana_x - self.box_x)**2 + (self.banana_y - self.box_y)**2) < 0.17:
            return self.reset()

        # range = [(80, 100), (170, 190)]
        range = [(40, 140), (220, 320)]
        chose_range = random.choice(range)
        self.banana.set_quat(Rotation.from_euler('xyz', [random.randint(chose_range[0], chose_range[1]), 0, 0], degrees=True).as_quat())
        # self.banana.set_quat(Rotation.from_euler('xyz', [random.randint(chose_range[0]), 0, 0], degrees=True).as_quat())
        # self.banana.set_quat(self.origin_euler)
        self.box.set_quat(Rotation.from_euler('xyz', [random.randint(0,360), 0, 90], degrees=True).as_quat())

        self.banana.set_pos(pos_banana)
        self.box.set_pos(pos_box)

        self.banana.morph.scale = random.uniform(0.9, 1.1)
        self.box.morph.scale = random.uniform(0.9, 1.1)

        self.step = 0
        self.scene.step()
        self.case += 1
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = get_args()
    collector = PickBanana(case=args.start,use_gs=args.use_gs,data_augmentation=args.data_augmentation,
                           save_dir=args.save_dir,reset_cam=args.reset_cam,single_view=args.single_view)
    collector.run(num_steps=args.num_steps)