from Gaussians import util_gau
import numpy as np
import torch
from Gaussians.renderer_ogl import GaussianRenderBase
from dataclasses import dataclass
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.io import read_image
from PIL import Image
import math

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class SSIM(nn.Module):
    def __init__(self, channel=1, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = self.create_window(window_size)

    def create_window(self, window_size):
        try:
            if hasattr(torch, "hanning_window"):
                _1D_window = torch.hanning_window(window_size).float()
            else:
                from scipy.signal import hann
                import numpy as np
                _1D_window = torch.from_numpy(np.array(hann(window_size))).float()
        except:
            window = 0.5 - 0.5 * torch.cos(2 * torch.pi * torch.arange(window_size).float() / (window_size - 1))
            _1D_window = window

        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(self.channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        if img1.size() != img2.size():
            raise ValueError("Input images must have the same dimensions")
        window = self.window.to(img1.device)
        # Calculate mean, variance and covariance
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_mu2 = mu1 * mu2
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

def psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]
    
def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus
    
def save_tensor_as_image(tensor, file_path):
    # 检查输入张量的形状
    if tensor.dim() != 3 or tensor.size(0) != 3:
        raise ValueError(f"Expected tensor of shape (3, H, W), got {tensor.shape}")
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        tensor = tensor.mul(255).clamp(0, 255).byte()
    elif tensor.dtype == torch.uint8:
        pass
    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
    
    image_array = tensor.permute(1, 2, 0).numpy()

    # (480, 640, 3)
    image = Image.fromarray(image_array)
    # image.save(file_path)
    # print(f"Image saved to {file_path}")

class CUDARenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        self.raster_settings = {
            "image_height": 480,
            "image_width": 640,
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "sh_degree": 3,
            "prefiltered": False,
            "debug": False
        }

        campos = torch.tensor([0.4349, 0.0012, 0.2371], device='cuda:0')
        viewmatrix = torch.tensor([[-0.0408, -0.8826, -0.4683,  0.0000],
        [-0.9933, -0.0148,  0.1144,  0.0000],
        [-0.1079,  0.4698, -0.8761,  0.0000],
        [ 0.0445,  0.2725,  0.4113,  1.0000]], device='cuda:0')

        projmatrix = torch.tensor([[-0.0306, -0.8826,  0.4684,  0.4683],
        [-0.7450, -0.0148, -0.1144, -0.1144],
        [-0.0809,  0.4698,  0.8763,  0.8761],
        [ 0.0334,  0.2725, -0.4314, -0.4113]], device='cuda:0')

        self.raster_settings['tanfovx'] = 1.333333391615188
        self.raster_settings['tanfovy'] = 1.000000043711391
        self.raster_settings['viewmatrix'] = viewmatrix
        self.raster_settings['projmatrix'] = projmatrix
        self.raster_settings['campos'] = campos

        self.update_gaussian_data()

    def update_gaussian_data(self):
        file_path = '/home/haoyu/code/GSim/exports/scene/scene-transform-wo-camera.ply'
        gaussians = util_gau.load_ply(file_path)
        self.need_rerender = True
        self.gaussians = gaus_cuda_from_cpu(gaussians)
        self.raster_settings["sh_degree"] = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1
        self.need_rerender = True

    def draw(self):
        self.need_rerender = False
        raster_settings = GaussianRasterizationSettings(**self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        with torch.no_grad():
            img, radii,_,_ = rasterizer(
                means3D = self.gaussians.xyz,
                means2D = None,
                shs = self.gaussians.sh,
                colors_precomp = None,
                opacities = self.gaussians.opacity,
                scales = self.gaussians.scale,
                rotations = self.gaussians.rot,
                cov3D_precomp = None
            )
        save_tensor_as_image(img, "output_image1.png")

# ===========================================================================
# ===========================================================================

class Renderer(GaussianRenderBase):
    def __init__(self):
        super().__init__()
        # self.update_gaussian_data()

    def update_gaussian_data(self,file_path='/home/haoyu/code/GSim/exports/scene/scene-transform-wo-camera.ply'):
        gaussians = util_gau.load_ply(file_path)
        self.gaussians = gaus_cuda_from_cpu(gaussians)

    def draw(self, raster_settings, theta, rho):
        raster_settings = GaussianRasterizationSettings(**raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        screenspace_points = (
        torch.zeros_like(
            self.gaussians.xyz, dtype=self.gaussians.xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0)
        img, _, _, _, _ = rasterizer(
            means3D = self.gaussians.xyz,
            means2D = screenspace_points,
            shs = self.gaussians.sh,
            colors_precomp = None,
            opacities = self.gaussians.opacity,
            scales = self.gaussians.scale,
            rotations = self.gaussians.rot,
            cov3D_precomp = None,
            theta = theta,
            rho = rho
        )
        # img, _,  = rasterizer(
        #     means3D = self.gaussians.xyz,
        #     means2D = None,
        #     shs = self.gaussians.sh,
        #     colors_precomp = None,
        #     opacities = self.gaussians.opacity,
        #     scales = self.gaussians.scale,
        #     rotations = self.gaussians.rot,
        #     cov3D_precomp = None,
        #     # theta = theta,
        #     # rho = rho
        # )
        return img

def main():
    device = torch.device("cuda:0")
    renderer = Renderer()
    real_image = read_image('/home/haoyu/code/GaussianSplattingViewer/GT.jpg').float() / 255.0
    real_image = real_image.to(device)

    # campos = torch.tensor([0.4349, 0.0012, 0.2371], device='cuda:0')
    # viewmatrix = torch.tensor([[-0.0408, -0.8826, -0.4683,  0.0000],
    #             [-0.9933, -0.0148,  0.1144,  0.0000],
    #             [-0.1079,  0.4698, -0.8761,  0.0000],
    #             [ 0.0445,  0.2725,  0.4113,  1.0000]], device='cuda:0')
    # campos = viewmatrix.inverse()[3, :3]
    # # print(f"Camera position: {campos}")
    # projmatrix = torch.tensor([[-0.0306, -0.8826,  0.4684,  0.4683],
    #             [-0.7450, -0.0148, -0.1144, -0.1144],
    #             [-0.0809,  0.4698,  0.8763,  0.8761],
    #             [ 0.0334,  0.2725, -0.4314, -0.4113]], device='cuda:0')
    # def full_proj_transform(self):
    #     return (
    #         self.world_view_transform.unsqueeze(0).bmm(
    #             self.projection_matrix.unsqueeze(0)
    #         )
    #     ).squeeze(0)
    # full_proj_transform = viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0)).squeeze(0)

    
    # raster_settings = {
    #     'image_height': 480, 'image_width': 640, 
    #     'tanfovx': 1.331000757969955, 'tanfovy': 1.000000043711391, 
    #     'bg': torch.tensor([0., 0., 0.], device='cuda:0'), 
    #     'scale_modifier': 1.0, 
    #     'viewmatrix': torch.tensor([[ 0.9952, -0.0338,  0.0923,  0.0000],
    #     [ 0.0000, -0.9391, -0.3436,  0.0000],
    #     [ 0.0982,  0.3420, -0.9346,  0.0000],
    #     [-0.5261, -0.2200,  0.8474,  1.0000]], device='cuda:0'), 
    #     'projmatrix': torch.tensor([[ 0.7477, -0.0338, -0.0923, -0.0923],
    #     [ 0.0000, -0.9391,  0.3437,  0.3436],
    #     [ 0.0738,  0.3420,  0.9347,  0.9346],
    #     [-0.3953, -0.2200, -0.8676, -0.8474]], device='cuda:0'), 
    #     'sh_degree': 3, 'campos': torch.tensor([0.4380, 0.0846, 0.9189], device='cuda:0'), 'prefiltered': False, 'debug': False}

    raster_settings = {
        'image_height': 480, 'image_width': 640, 
        'tanfovx': 0.7548988664410912, 'tanfovy': 0.5661741498308184, 
        'bg': torch.tensor([0., 0., 0.], device='cuda:0'), 
        'scale_modifier': 1.0, 
        'viewmatrix': torch.tensor([[ 0.0692, -0.9059, -0.4177,  0.0000],
        [-0.9969, -0.0467, -0.0638,  0.0000],
        [ 0.0383,  0.4209, -0.9063,  0.0000],
        [-0.0453,  0.2892,  0.4494,  1.0000]], device='cuda:0'), 
        'projmatrix': torch.tensor([[ 0.0519, -0.9059,  0.4178,  0.4177],
        [-0.7476, -0.0467,  0.0638,  0.0638],
        [ 0.0287,  0.4209,  0.9065,  0.9063],
        [-0.0340,  0.2892, -0.4695, -0.4494]], device='cuda:0'), 
        'sh_degree': 3, 'campos': torch.tensor([ 0.4529, -0.0030,  0.2873], device='cuda:0'), 'prefiltered': False, 'debug': False}
    # viewmatrix = raster_settings['viewmatrix']
    
    print('raster_settings raw:', raster_settings)
    raster_settings = {
        'image_height': 480, 'image_width': 640, 
        'tanfovx': 0.7548988664410912, 'tanfovy': 0.5661741498308184, 
        'bg': torch.tensor([0., 0., 0.], device='cuda:0'), 'scale_modifier': 1.0, 
        'viewmatrix': torch.tensor([[ 0.0366,  0.7512, -0.6590,  0.0000],
        [ 0.9890,  0.0672,  0.1315,  0.0000],
        [ 0.1431, -0.6566, -0.7405,  0.0000],
        [ 0.0150, -0.1763,  0.7723,  1.0000]], device='cuda:0'), 
        'projmatrix': torch.tensor([[ 0.0485,  1.3268, -0.6592, -0.6590],
        [ 1.3102,  0.1187,  0.1315,  0.1315],
        [ 0.1895, -1.1597, -0.7407, -0.7405],
        [ 0.0199, -0.3114,  0.7524,  0.7723]], device='cuda:0'), 'sh_degree': 3, 
        'campos': torch.tensor([ 0.6408, -0.1046,  0.4540], device='cuda:0'), 'prefiltered': False, 'debug': False}
    
    z_near = 0.01
    z_far = 100.0
    fov_x = 2 * torch.atan(torch.tensor(raster_settings['tanfovx']))
    fov_y = 2 * torch.atan(torch.tensor(raster_settings['tanfovy']))
    fov_x = 1.2929323299305089
    fov_y = 1.0303522473923061
    print('fov_x:', fov_x, 'fov_y:', fov_y)
    projmatrix_raw = getProjectionMatrix(z_near, z_far, fov_x, fov_y).transpose(0, 1).cuda()
    # print('projmatrix_raw1:', projmatrix_raw)
    # projmatrix_raw = torch.matmul(torch.linalg.pinv(raster_settings['viewmatrix']), raster_settings['projmatrix'])
    # print('projmatrix_raw2:', projmatrix_raw)
    raster_settings['projmatrix_raw'] = projmatrix_raw
    # projmatrix = (viewmatrix.unsqueeze(0).bmm(projmatrix_raw.unsqueeze(0))).squeeze(0)
    # campos = viewmatrix.inverse()[3, :3]
    # raster_settings['projmatrix'] = projmatrix
    # raster_settings['campos'] = campos
    # raster_settings['viewmatrix'] = viewmatrix
    # raster_settings['projmatrix'] = viewmatrix.unsqueeze(0).bmm(projmatrix_raw.unsqueeze(0)).squeeze(0)
    print('raster_settings wh:', raster_settings)
    # add noise
    # viewmatrix = raster_settings['viewmatrix']
    # c2w = torch.inverse(viewmatrix.T)
    # c2w[:3, 3] = c2w[:3, 3] + torch.randn(3, device=device) * 0.1
    # viewmatrix = torch.inverse(c2w).T 
    # campos = viewmatrix.inverse()[3, :3]
    # raster_settings['campos'] = campos
    # raster_settings['viewmatrix'] = viewmatrix
    # raster_settings['projmatrix'] = viewmatrix.unsqueeze(0).bmm(projmatrix_raw.unsqueeze(0)).squeeze(0)
    # add noise
    # raster_settings['projmatrix_raw'] = projmatrix_raw
    # projmatrix = viewmatrix.unsqueeze(0).bmm(projmatrix_raw.unsqueeze(0)).squeeze(0)
    # raster_settings['projmatrix_raw'] = torch.matmul(torch.linalg.pinv(viewmatrix), projmatrix)
    print('raster_settings', raster_settings)
    
    
    # tanfovx = torch.tensor(1.333333391615188, device=device)
    # tanfovy = torch.tensor(1.000000043711391, device=device)
    
    theta = nn.Parameter(torch.tensor([[0, 0, 0]], device=device).float(), requires_grad=True)  # Quaternion in wxyz format
    rho = nn.Parameter(torch.tensor([[0, 0, 0]], device=device).float(), requires_grad=True)
    # optimizer = optim.Adam([theta, rho], lr=0.003)
    optimizer = optim.Adam([
        {'params': theta, 'lr': 0.0001},
        {'params': rho, 'lr': 0.0001},
    ], lr=0.003)
    renderer.gaussians.xyz = renderer.gaussians.xyz.detach().clone()
    renderer.gaussians.rot = renderer.gaussians.rot.detach().clone()
    renderer.gaussians.scale = renderer.gaussians.scale.detach().clone()
    renderer.gaussians.opacity = renderer.gaussians.opacity.detach().clone()
    renderer.gaussians.sh = renderer.gaussians.sh.detach().clone()
    # renderer.gaussians.xyz.requires_grad_(True)
    # renderer.gaussians.sh.requires_grad_(True)
    # renderer.gaussians.opacity.requires_grad_(True)
    # renderer.gaussians.scale.requires_grad_(True)
    # renderer.gaussians.rot.requires_grad_(True)

    ssim = SSIM(channel=3, window_size=11, size_average=True).to(device)
    best_loss = 99
    for iteration in range(10000):
        print(f"Iteration {iteration + 1}/10000")
        optimizer.zero_grad()

        # raster_settings = {
        #     "image_height": 480,
        #     "image_width": 640,
        #     "bg": torch.Tensor([0., 0., 0]).float().cuda(),
        #     "scale_modifier": 1.,
        #     "sh_degree": 3,
        #     "prefiltered": False,
        #     "debug": False,
        #     "campos": campos,
        #     "viewmatrix": viewmatrix,
        #     "projmatrix": full_proj_transform,
        #     "projmatrix_raw": projmatrix,
        #     "tanfovx": tanfovx,
        #     "tanfovy": tanfovy,
        # }
        raster_settings = raster_settings

        rendered_image = renderer.draw(raster_settings, theta, rho)
        
        # resize real image to match rendered image
        if rendered_image.shape[1:] != real_image.shape[1:]:
            real_image = F.interpolate(real_image.unsqueeze(0), size=rendered_image.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        loss_l1 = torch.nn.functional.l1_loss(rendered_image, real_image)
        loss_l1 = (torch.abs(rendered_image - real_image)).mean()
        loss_ssim = 1 - ssim(rendered_image.unsqueeze(0), real_image.unsqueeze(0))

        # if iteration < 200:
        #     loss = loss_l1 + 0.8 * loss_ssim * 0.2
        # elif iteration < 500:
        #     loss = loss_l1 + 0.5 * loss_ssim * 0.4 
        # else:
        #     loss = loss_l1 + 0.3 * loss_ssim * 0.4
        loss = loss_l1 + 0.8 * loss_ssim * 0.2

        # save_tensor_as_image(rendered_image.detach(), f"saved_img/iter_{iteration}.png")
        # print('theta:',theta, 'rho:', rho)
        print('loss:', loss.item())       
        loss.backward()

        if(loss.item() < best_loss):
            best_loss = loss.item()
            best_img = rendered_image.detach().clone()
            best_theta = theta.detach().clone()
            best_rho = rho.detach().clone()

        renderer.gaussians.xyz.grad = None
        renderer.gaussians.sh.grad = None
        renderer.gaussians.opacity.grad = None
        renderer.gaussians.scale.grad = None
        renderer.gaussians.rot.grad = None
        optimizer.step()
        # print('theta:', theta)
        # print('rho:', rho)
        from pose_utils import SE3_exp
        converged_threshold = 1e-4
        tau = torch.cat([rho, theta], axis=0)
        # print('tau:', tau)
        # T_w2c = torch.eye(4, device=tau.device)
        T_w2c = raster_settings['viewmatrix'].T
        # print('T_w2c:', T_w2c)


        new_w2c = SE3_exp(tau) @ T_w2c

        viewmatrix = new_w2c.T
        projmatrix_raw = raster_settings['projmatrix_raw']
        projmatrix = viewmatrix.unsqueeze(0).bmm(
            projmatrix_raw.unsqueeze(0)
        ).squeeze(0)
        
        raster_settings['viewmatrix'] = viewmatrix
        raster_settings['projmatrix'] = projmatrix
        raster_settings['campos'] = viewmatrix.inverse()[3, :3]
        # print('raster_settings in iteration', raster_settings)
        converged = tau.norm() < converged_threshold
        if converged:
            print(f"Converged at iteration {iteration + 1}")
            break
        

        theta.data.fill_(0)
        rho.data.fill_(0)
        if iteration % 50 == 0:
            print(f"Iter {iteration}, Loss: {loss.item():.6f}")
            save_tensor_as_image(rendered_image.detach(), f"saved_img/iter_{iteration}.png")
    
    print(f"New best loss: {best_loss:.6f}")
    print(f"Best theta: {best_theta}")
    print(f"Best rho: {best_rho}")
    save_tensor_as_image(rendered_image.detach(), f"saved_img/best_iter_{iteration}_theta{best_theta.data}_rho{best_rho.data}.png")


if __name__ == "__main__":
    main()