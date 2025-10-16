#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test URDF in PyBullet simulation"""

import sys
import time
import math


def test_urdf(urdf_path: str):
    """Load and test URDF in PyBullet"""
    
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("Error: PyBullet not installed")
        print("Install with: pip install pybullet")
        return 1
    
    print(f"Testing URDF: {urdf_path}")
    
    # Connect to physics engine
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, -9.81, 0)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Load URDF
    try:
        robotId = p.loadURDF(urdf_path, [0, 0, 0.5])
    except Exception as e:
        print(f"Error loading URDF: {e}")
        p.disconnect()
        return 1
    
    # Get joint info
    num_joints = p.getNumJoints(robotId)
    print(f"Robot has {num_joints} joint(s)")
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        print(f"  Joint {i}: {joint_info[1].decode('utf-8')} (type: {joint_info[2]})")
    
    # Add slider for joint control
    if num_joints > 0:
        slider = p.addUserDebugParameter("Joint Angle (deg)", 0, 90, 0)
    
    print("\nSimulation running. Close window to exit.")
    print("Use slider to control joint angle")
    
    try:
        while True:
            if num_joints > 0:
                angle_deg = p.readUserDebugParameter(slider)
                angle_rad = math.radians(angle_deg)
                p.setJointMotorControl2(
                    robotId, 0,
                    p.POSITION_CONTROL,
                    targetPosition=angle_rad,
                    force=100
                )
            
            p.stepSimulation()
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("\nStopping simulation")
    
    p.disconnect()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.test_pybullet <urdf_file>")
        sys.exit(1)
    
    sys.exit(test_urdf(sys.argv[1]))


