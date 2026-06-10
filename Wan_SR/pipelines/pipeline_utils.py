import math
import os

import cv2
import torch

import torch.nn.functional as F

from typing import List, Tuple
import imageio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


WAN_T2V_1_3B_FILES = (
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)


def resolve_wan_model_paths(wan_model_dir: str) -> List[str]:
    if not wan_model_dir:
        raise ValueError("--wan_model_dir must not be empty.")
    return [os.path.join(wan_model_dir, file_name) for file_name in WAN_T2V_1_3B_FILES]



def pad_video_auto(video: torch.Tensor):
    b, c, t, h, w = video.shape

    # -----------------------------
    # 1) time -> 4*n+1
    # -----------------------------
    n = math.ceil((t - 1) / 4)
    target_t = 4 * n + 1

    if t < target_t:
        pad_t = target_t - t
        last_frame = video[:, :, -1:, :, :].expand(b, c, pad_t, h, w)
        video = torch.cat([video, last_frame], dim=2)

    # -----------------------------
    # 2) width/height -> 16*n
    # -----------------------------
    target_h = math.ceil(h / 16) * 16
    target_w = math.ceil(w / 16) * 16

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h > 0 or pad_w > 0:
        video = F.pad(video.squeeze(0), (0, pad_w, 0, pad_h), mode="replicate").unsqueeze(0)

    return video, t, target_t, h, target_h, w, target_w


def aggregate_caption(captions, b, ram_sample):
    video_obj_sets = [set() for _ in range(b)]

    for idx, s in enumerate(captions):
        video_idx = idx // ram_sample
        video_obj_sets[video_idx].update(s.split(", "))

    return [", ".join(list(objs)) for objs in video_obj_sets]


def paired_video_to_chunks(pred: torch.Tensor, gt: torch.Tensor, patch_size=224):
    assert pred.shape == gt.shape, "Pred and GT must have the same shape"
    B, C, T, H, W = pred.shape

    def compute_positions(size, patch_size):
        if size <= patch_size:
            return [0]
        n = (size + patch_size - 1) // patch_size
        stride = (size - patch_size) // (n - 1) if n > 1 else patch_size
        return [i * stride for i in range(n-1)] + [size - patch_size]

    h_positions = compute_positions(H, patch_size)
    w_positions = compute_positions(W, patch_size)

    pred_patches, gt_patches = [], []
    for i in h_positions:
        for j in w_positions:
            pred_patch = pred[:, :, :, i:i+patch_size, j:j+patch_size]  # (B,C,T,224,224)
            gt_patch = gt[:, :, :, i:i+patch_size, j:j+patch_size]
            pred_patches.append(pred_patch)
            gt_patches.append(gt_patch)

    pred_patches = torch.cat(pred_patches, dim=0)  # (B*nH*nW, C, T, 224, 224)
    gt_patches = torch.cat(gt_patches, dim=0)
    return pred_patches, gt_patches

def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params, macro_block_size=1)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

def expand_patch_embedding(old_conv: nn.Conv3d, factor: int = 2):
    assert isinstance(old_conv, nn.Conv3d)
    old_in = old_conv.in_channels
    out_ch = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    dilation = old_conv.dilation
    groups = old_conv.groups
    has_bias = old_conv.bias is not None

    new_in = old_in * factor
    new_conv = nn.Conv3d(
        in_channels=new_in,
        out_channels=out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=has_bias,
    )

    new_weight = torch.zeros((out_ch, new_in, *kernel_size), dtype=old_conv.weight.dtype, device=old_conv.weight.device)
    new_weight[:, :old_in, ...] = old_conv.weight.data.clone()
    new_conv.weight = nn.Parameter(new_weight, requires_grad=old_conv.weight.requires_grad)

    if has_bias:
        new_conv.bias = nn.Parameter(old_conv.bias.data.clone(), requires_grad=old_conv.bias.requires_grad)

    return new_conv

def get_torch_dtype(dtype_str: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]

def open_video_writer(save_path: str, fps: float, quality: int = 9):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return imageio.get_writer(
        save_path,
        fps=fps,
        quality=quality,
        macro_block_size=1,
    )

def probe_video_meta(input_path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size: {input_path}")
    if fps <= 0:
        fps = 15.0
    if num_frames <= 0:
        raise RuntimeError(f"Invalid frame count: {input_path}")

    return width, height, fps, num_frames
