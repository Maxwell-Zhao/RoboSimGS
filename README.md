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
- [✅] Release arXiv technique report
- [✅] Release 3D reconstruction pipeline
- [✅] Release MLLM-based Articulation Inference and Physics Estimation
- [ ] Release simulated data generation pipeline
- [ ] Release codebase for training Diffusion Policy on our simulated data
- [ ] Sim2Real demployment pipeline


## 📚 Table of Contents

1. **[Overview](#overview)**  
   - Links: [Website](https://robosimgs.github.io/) • [Arxiv]() • [Video](https://www.youtube.com/watch?v=nvUXAovzc6Q)
 

2. **[Installation & Setup](#installation)**  


3. **[Scene Reconstruction](#scene-reconstruction)**  
    3.1 [Background Reconstruction](#background-reconstruction)
    
    3.2 [Object Reconstruction](#object-reconstruction)

   - 3.2.1 [Articulation Inference](Articulation/Articulation.md)

   - 3.2.2 [Physics Estimation](Articulation/Articulation.md)


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
We reconstruct a 3D background scene using the 3D Gaussian Splatting (3DGS) method within the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). For more in-depth information, always refer to the official Nerfstudio GitHub repository. The final output of this process will be a .ply file representing the 3D Gaussian Splatting scene, which can be viewed in compatible real-time renderers.

## Object Reconstruction
As mentioned in our paper, the object reconstruction is performed using the 'AR Code' app on an iPhone 16 Pro. By scanning the object with the phone, we can obtain a high-quality mesh.

**MLLM-driven Articulation Inference**. A comprehensive overview of the architecture is available in the [Document](Articulation/Articulation.md).

**MLLM-driven Physics Estimation**. A comprehensive overview of the architecture is available in the [Document](Articulation/Articulation.md).

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
