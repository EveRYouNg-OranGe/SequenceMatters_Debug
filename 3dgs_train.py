#!/usr/bin/env python
# coding: utf-8

# # 3DGS训练过程拆解
# 
# 需要提前完成VSR图片超分，得到HR图片集作为高斯模型训练集

# ## 参数导入

# In[1]:


import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui 
import sys
from scene import Scene, GaussianModel 
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import shutil
from utils.general_utils import load_config
from vsr.utils_vsr import (
    setup_paths_and_params,
    load_images,
    load_vsr_model,
    process_S,
    process_ALS,
    create_video_from_images,
)


# In[2]:


# Set up command line argument parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)

parser.add_argument("--config", type=str, default=None, help="Path to configuration YAML file")

# -m : args.model_path, 3d模型保存路径，最后一个子文件夹必须为场景名，用来解析识别场景
args = parser.parse_args([
    "-m", "../OUTPUTS/3DGS/ship",
    "--eval", 
    "--config", "configs/blender.yml"
])
args.save_iterations.append(args.iterations)

print(f"\nGet args: {args}\n")

args = load_config(args)
print("Optimizing " + args.model_path)
# 遍历并打印
for key, value in vars(args).items():
    print(f"{key:<25}: {value}")

# Initialize system state (RNG)
# safe_state(args.quiet)

# Start GUI server, configure and run training
# network_gui.init(args.ip, args.port)
# torch.autograd.set_detect_anomaly(args.detect_anomaly)


# ## 模型训练

# In[3]:


''' 训练所需参数 ''' 
lambda_tex=0.40
subpixel="avg"
(
    dataset, opt, pipe, testing_iterations, saving_iterations, 
    checkpoint_iterations, checkpoint, debug_from, lambda_tex, subpixel
) = (
    lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, 
    args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.lambda_tex, 
    args.subpixel
)


# In[4]:


''' 1. 准备日志和输出目录 '''
first_iter = 0

# args.model_path即3d高斯模型存储路径
if not args.model_path:
    # 调用环境变量OAR_JOB_ID， 变量通常用于集群或调度系统（如 OAR、SLURM）中，用来表示任务的唯一编号； 
    # 如果不是这种环境， 那就用随机字符串的前十位；
    if os.getenv('OAR_JOB_ID'):
        unique_str=os.getenv('OAR_JOB_ID')
    else:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

# Set up output folder
print("Output folder: {}".format(args.model_path))
os.makedirs(args.model_path, exist_ok = True)
# 生成一个cfg文件，写入参数
with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
    cfg_log_f.write(str(Namespace(**vars(args))))

# Create Tensorboard writer
tb_writer = None
if TENSORBOARD_FOUND:
    tb_writer = SummaryWriter(args.model_path)
else:
    print("Tensorboard not available: not logging progress")

# In[5]:


''' 2. 初始化高斯模型 '''
gaussians = GaussianModel(dataset.sh_degree)


# In[6]:


''' 3. 创建场景（加载训练相机）'''
scene = Scene(dataset, gaussians)


# In[7]:


''' 4. 设置优化器 '''
gaussians.training_setup(opt)


# In[8]:


''' 5. 恢复检查点（如果提供） '''
if checkpoint:
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)


# In[9]:


''' 训练实时数据报告 （删除了tensorboard相关代码） '''
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs):

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        
        torch.cuda.empty_cache()


# In[10]:


# 初始化一个背景色tensor
bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
'''
>>background
tensor([0., 0., 0.], device='cuda:0')
'''


# In[11]:


# 计时与同步工具
iter_start = torch.cuda.Event(enable_timing = True)
iter_end = torch.cuda.Event(enable_timing = True)


# In[12]:


viewpoint_stack = None
# 定义了一个二维平均池化层
'''用于对输入特征图（如图片或卷积层输出）进行降采样。 它会将输入划分成若干个小区域（称为窗口或 kernel）， 然后在每个区域中计算所有像素的平均值，
从而实现： 降低分辨率、减少计算量、保留整体特征趋势。'''
avg_kernel = torch.nn.AvgPool2d(4, stride=4)  


# In[13]:


ema_loss_for_log = 0.0
progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
first_iter += 1
for iteration in range(first_iter, opt.iterations + 1):        
    # if network_gui.conn == None:
    #     network_gui.try_connect()
    # while network_gui.conn != None:
    #     try:
    #         net_image_bytes = None
    #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
    #         if custom_cam != None:
    #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
    #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
    #         network_gui.send(net_image_bytes, dataset.source_path)
    #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
    #             break
    #     except Exception as e:
    #         network_gui.conn = None

    iter_start.record()  # 开始计时 

    gaussians.update_learning_rate(iteration) # 更新学习率

    # Every 1000 its we increase the levels of SH up to a maximum degree
    # 每迭代1000次， 提高SH水平， 直到最高水平
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree() # active_sh_degree += 1
    '''每1000次迭代增加球谐函数的度数; 球谐函数用于表示外观变化，更高度数能捕捉更复杂的细节'''
        
    ### HR scale
    # Pick a random Camera 选择一个相机
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    idx_cam = randint(0, len(viewpoint_stack)-1)
    viewpoint_cam = viewpoint_stack.pop(idx_cam)

    # Render
    # 在迭代次数到达一定数值时开启debug模式
    if (iteration - 1) == debug_from:
        pipe.debug = True

    bg = torch.rand((3), device="cuda") if opt.random_background else background # 随机背景或者固定背景 （为什么需要随机背景？）
    # 高分辨率渲染
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    # Loss 高分辨率损失函数
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss_tex = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    ### LR scale
    # Pick a random Camera
    # 高分辨率渲染结果降采样到低分辨率； 使用平均池化或者双三次插值
    if subpixel == 'avg':
        image_avg = avg_kernel(image)
    elif subpixel == 'bicubic':
        image_avg = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=0.25, mode='bicubic', antialias=True).squeeze(0)
    else:
        raise Exception("Wrong sub-pixel option")

    gt_image_lr = viewpoint_cam.original_image_lr.cuda()
    # 确保低分辨率GT图像与下采样结果尺寸匹配
    if image_avg.shape != gt_image_lr.shape: 
        # import torch.nn.functional as F
        gt_image_lr = torch.nn.functional.interpolate(gt_image.unsqueeze(0), size=image_avg.size()[-2:], mode='bicubic', antialias=True).squeeze(0)

    # Loss 计算低分辨率损失函数
    Ll1_sp = l1_loss(image_avg, gt_image_lr)
    loss_sp = (1.0 - opt.lambda_dssim) * Ll1_sp + opt.lambda_dssim * (1.0 - ssim(image_avg, gt_image_lr))

#     if iteration == opt.iterations - 5000:
#         import torchvision.transforms as transforms
#         from PIL import Image

#         to_pil_image = transforms.ToPILImage()

#         gt_image_lr_pil = to_pil_image(gt_image_lr)
#         gt_image_lr_pil.save("gt_image_lr_pil.png")

#         image_avg_pil  = to_pil_image(image_avg)
#         image_avg_pil.save("image_avg_pil.png")

    # 最终损失计算和反向传播
    lambda_tex_scheduled = lambda_tex
    loss = (1.0 - lambda_tex_scheduled) * loss_sp + lambda_tex_scheduled * loss_tex
    loss.backward()

    iter_end.record()

    # 由高斯点云渲染3D模型
    with torch.no_grad():
        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log # EMA损失：使用0.4/0.6权重计算平滑损失，避免波动
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

        # Log and save 记录损失、渲染时间等指标，可能包括测试集评估
        training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
        # 保存设定的迭代点处的模型
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        # Densification
        # 密度控制； 在细节不足或误差大的地方自动“加点”； 在冗余或不可见区域“删点”。
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()

        # Optimizer step 优化
        if iteration < opt.iterations:
            gaussians.optimizer.step() # 执行梯度下降
            gaussians.optimizer.zero_grad(set_to_none = True) # 为下一次迭代准备，set_to_none=True节省内存

        # 输出保存高精度3D Gaussian点云模型
        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


# In[ ]:




