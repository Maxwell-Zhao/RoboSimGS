#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""URDF file builder"""

import numpy as np
from typing import List


class URDFBuilder:
    """Build URDF XML from components"""
    
    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        self.links = []
        self.joints = []
    
    def add_link(
        self,
        name: str,
        mesh_file: str,
        color: List[float] = [0.8, 0.8, 0.8, 1.0],
        mass: float = 1.0,
        inertia: List[float] = None
    ):
        """Add a link to the robot"""
        
        if inertia is None:
            inertia = [0.01, 0.0, 0.0, 0.01, 0.0, 0.01]
        
        link_xml = f'''  <link name="{name}">
    <visual>
      <geometry>
        <mesh filename="{mesh_file}" scale="1 1 1"/>
      </geometry>
      <material name="{name}_material">
        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{mesh_file}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{inertia[0]}" ixy="{inertia[1]}" ixz="{inertia[2]}" 
               iyy="{inertia[3]}" iyz="{inertia[4]}" izz="{inertia[5]}"/>
    </inertial>
  </link>'''
        
        self.links.append(link_xml)
    
    def add_revolute_joint(
        self,
        name: str,
        parent: str,
        child: str,
        origin_xyz: List[float],
        origin_rpy: List[float],
        axis: np.ndarray,
        limit_lower: float,
        limit_upper: float,
        effort: float = 10.0,
        velocity: float = 1.0,
        friction: float = 0.5,
        damping: float = 0.1
    ):
        """Add a revolute joint"""
        
        joint_xml = f'''  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{origin_xyz[0]:.6f} {origin_xyz[1]:.6f} {origin_xyz[2]:.6f}" 
            rpy="{origin_rpy[0]:.6f} {origin_rpy[1]:.6f} {origin_rpy[2]:.6f}"/>
    <axis xyz="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}"/>
    <limit lower="{limit_lower:.6f}" upper="{limit_upper:.6f}" 
           effort="{effort:.2f}" velocity="{velocity:.2f}"/>
    <dynamics friction="{friction:.2f}" damping="{damping:.2f}"/>
  </joint>'''
        
        self.joints.append(joint_xml)
    
    def add_fixed_joint(
        self,
        name: str,
        parent: str,
        child: str,
        origin_xyz: List[float] = [0, 0, 0],
        origin_rpy: List[float] = [0, 0, 0]
    ):
        """Add a fixed joint"""
        
        joint_xml = f'''  <joint name="{name}" type="fixed">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{origin_xyz[0]:.6f} {origin_xyz[1]:.6f} {origin_xyz[2]:.6f}" 
            rpy="{origin_rpy[0]:.6f} {origin_rpy[1]:.6f} {origin_rpy[2]:.6f}"/>
  </joint>'''
        
        self.joints.append(joint_xml)
    
    def save(self, filepath: str):
        """Save URDF to file"""
        
        urdf_content = f'''<?xml version="1.0"?>
<robot name="{self.robot_name}">

{chr(10).join(self.links)}

{chr(10).join(self.joints)}

</robot>'''
        
        with open(filepath, 'w') as f:
            f.write(urdf_content)


