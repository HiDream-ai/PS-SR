import os
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F
from PIL import Image

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
# from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from .degradation_video import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)

cur_path = os.path.dirname(os.path.abspath(__file__))


def filter3D(video, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        video (Tensor): (b, c, t, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, t, h, w = video.size()
    if k % 2 == 1:
        pad = k // 2
        # reshape video to (b*t, c, h, w) for 2D filtering
        video = video.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        # pad video
        video = F.pad(video, (pad, pad, pad, pad), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    # reshape for batch and time combined convolution
    ph, pw = video.size()[-2:]
    if kernel.size(0) == 1:
        # 使用相同 kernel 处理所有 batch 和 time 的帧
        video = video.view(b * t * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        output = F.conv2d(video, kernel, padding=0)
    else:
        # 每个 batch 使用不同的 kernel
        video = video.view(1, b * t * c, ph, pw)
        kernel = kernel.unsqueeze(1).repeat(
            1, c, 1, 1).view(b * t * c, 1, k, k)
        output = F.conv2d(video, kernel, groups=b * t * c)

    # reshape back to (b, c, t, h, w)
    output = output.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return output


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        # ignore_security_alert_wait_for_fix RCE
        opt = yaml.load(f, Loader=Loader)

    return opt


class RealESRGAN_video_degradation(object):
    def __init__(self, opt_name=None, device='cpu'):
        opt_path = f'{cur_path}/{opt_name}'
        self.opt = opt_parse(opt_path)
        self.device = device  # torch.device('cpu')
        optk = self.opt['kernel_info']

        # blur settings for the first degradation
        self.blur_kernel_size = optk['blur_kernel_size']
        self.kernel_list = optk['kernel_list']
        self.kernel_prob = optk['kernel_prob']
        self.blur_sigma = optk['blur_sigma']
        self.betag_range = optk['betag_range']
        self.betap_range = optk['betap_range']
        self.sinc_prob = optk['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = optk['blur_kernel_size2']
        self.kernel_list2 = optk['kernel_list2']
        self.kernel_prob2 = optk['kernel_prob2']
        self.blur_sigma2 = optk['blur_sigma2']
        self.betag_range2 = optk['betag_range2']
        self.betap_range2 = optk['betap_range2']
        self.sinc_prob2 = optk['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = optk['final_sinc_prob']

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_shaper = USMSharp().to(self.device)

    def color_jitter_pt(self, img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(
                    brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(
                    contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(
                    saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def random_augment(self, img_gt_list):

        # random horizontal flip
        img_gt_list, status = augment(
            img_gt_list, hflip=True, rotation=False, return_status=True)
        """
        # random color jitter 
        if np.random.uniform() < self.opt['color_jitter_prob']:
            jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.opt['gray_prob']:
            #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        """
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_list = img2tensor(img_gt_list, bgr2rgb=False, float32=True)
        video = torch.stack(img_gt_list, dim=1).unsqueeze(0)
        return video

    def random_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return kernel, kernel2, sinc_kernel

    @torch.no_grad()
    def degrade_process(self, video_gt, resize_bak=True):
        # img_gt = self.random_augment(img_gt)
        # kernel1, kernel2, sinc_kernel = self.random_kernels()

        T = video_gt.shape[2]

        kernel1_list = []
        kernel2_list = []
        sinc_kernel_list = []

        for _ in range(T):
            kernel1, kernel2, sinc_kernel = self.random_kernels()
            kernel1_list.append(kernel1)
            kernel2_list.append(kernel2)
            sinc_kernel_list.append(sinc_kernel)

        kernel1 = torch.stack(kernel1_list, dim=0).to(self.device)         # [T, 21, 21]
        kernel2 = torch.stack(kernel2_list, dim=0).to(self.device)         # [T, 21, 21]
        sinc_kernel = torch.stack(sinc_kernel_list, dim=0).to(self.device) # [T, 21, 21]

        video_gt, kernel1, kernel2, sinc_kernel = video_gt.to(self.device), kernel1.to(
            self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        # img_gt = self.usm_shaper(img_gt) # shaper gt
        ori_h, ori_w = video_gt.size()[3:5]

        # scale_final = random.randint(4, 16)
        scale_final = 4

        b, c, t, h, w = video_gt.shape
        out = video_gt.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)

        # ----------------------- The first degradation process ----------------------- #
        # blur
        # out = filter2D(out, kernel1.unsqueeze(0))
        out = filter2D(out, kernel1)
        # random resize
        # updown_type = random.choices(
        #     ['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        updown_type = 'down'
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # scale = 0.17
        # mode = 'bicubic'

        # out = F.interpolate(out, scale_factor=scale, mode=mode)

        # noise
        gray_noise_prob = self.opt['gray_noise_prob']
        # if np.random.uniform() < self.opt['gaussian_noise_prob']:
        if True:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        # jpeg_p = out[0:1, :, :, :].new_zeros(
        #     out[0:1, :, :, :].size(0)).uniform_(*self.opt['jpeg_range'])
        # jpeg_p = jpeg_p.repeat(out.size(0))

        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        # if np.random.uniform() < self.opt['second_blur_prob']:
        if True:
            # out = filter2D(out, kernel2.unsqueeze(0))
            out = filter2D(out, kernel2)
        # random resize
        # updown_type = random.choices(
        #     ['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        updown_type = 'keep'
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob2']
        # if np.random.uniform() < self.opt['gaussian_noise_prob2']:
        if True:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        # if np.random.uniform() < 0.5:
        if True:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(
                ori_h // scale_final, ori_w // scale_final), mode=mode)
            # out = filter2D(out, sinc_kernel.unsqueeze(0))
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            # jpeg_p = out.new_zeros(out.size(0)).uniform_(
            #     *self.opt['jpeg_range2'])
            jpeg_p = out[0:1, :, :, :].new_zeros(
                out[0:1, :, :, :].size(0)).uniform_(*self.opt['jpeg_range2'])
            jpeg_p = jpeg_p.repeat(out.size(0))
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(
                ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel.unsqueeze(0))

        if np.random.uniform() < self.opt['gray_prob']:
            out = rgb_to_grayscale(out, num_output_channels=1)

        if np.random.uniform() < self.opt['color_jitter_prob']:
        # if True:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = self.color_jitter_pt(
                out, brightness, contrast, saturation, hue)

        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)

        out = out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        # clamp and round
        video_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return video_gt, video_lq

    def degrade_process_video(self, video_gt_dir, resize_bak=False):
        img_gt_list = sorted(glob.glob(video_gt_dir + '/*.png'))
        img_gt_ndarray_list = []
        for img_gt_path in img_gt_list:
            gt_img = Image.open(img_gt_path).convert('RGB')
            img_gt_ndarray_list.append(np.asarray(gt_img)/255.)
        video_gt = self.random_augment(img_gt_ndarray_list)
        video_gt, video_lq = self.degrade_process(video_gt, resize_bak)
        return video_gt, video_lq
    
