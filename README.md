<h1 align="center"> High-Fidelity Simulated Data Generation for Real-World 

Zero-Shot Robotic Manipulation Learning 

with Gaussian Splatting </h1>

<div align="center">

[[Website]](https://robosimgs.github.io/)
[[Arxiv]]()
[[Video]](https://www.youtube.com/watch?v=nvUXAovzc6Q)

[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

<img src="imgs/demo.gif" width="400px"/>

</div>

## Pipeline

<div align="center">
<img src="imgs/pipeline.png" width="520px"/>
</div>

## TODO
- [] Release 3D reconstruction pipeline
- [] Release MLLM-based Articulation Inference and Physics Estimation
- [] Release simulated data generation pipeline
- [] Release codebase for training Diffusion Policy on our simulated data
- [] Sim2Real demployment pipeline


## 📚 Table of Contents

1. **[Overview](#overview)**  
   - Links: [Website](https://robosimgs.github.io/) • [Arxiv]() • [Video](https://www.youtube.com/watch?v=nvUXAovzc6Q)
 

2. **[Installation & Setup](#installation)**  


3. **[Scene Reconstruction](#scene-reconstruction)**  
   3.1 [Background Reconstruction](#background-reconstruction)  
   3.2 [Object Reconstruction](#object-reconstruction)  


4. **[Simulated Data Generation](#simulated-data-generation)**  
   4.1 World Coordinate Frame Alignment  
   4.2 Camera Pose Alignment
   4.3 Data Generation


5. **[Policy Training](#policy-training)**  

6. **[Sim2Real Deployment](#sim2real-deployment)**  

7. **[Citation](#citation)**  

8. **[License](#license)**


# Installation

RoboSimGS codebase is built on top of [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) and [Lerobot](https://github.com/huggingface/lerobot).  We are actively working on this section and will update it soon. 🚧


# Scene Reconstruction
## Background Reconstruction
We are actively working on this section and will update it soon. 🚧

## Object Reconstruction
As mentioned in our paper, the object reconstruction is performed using the 'AR Code' app on an iPhone 16 Pro. By scanning the object with the phone, we can obtain a high-quality mesh.

# Simulated Data Generation
We are actively working on this section and will update it soon. 🚧

# Policy Training
We are actively working on this section and will update it soon. 🚧


# Sim2Real Deployment
We are actively working on this section and will update it soon. 🚧


# Citation
If you find our work useful, please consider citing us!

```bibtex
@article{he2025asap,
  title={High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting},
  author={},
  journal={},
  year={2025}
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
