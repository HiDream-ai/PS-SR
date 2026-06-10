from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Tuple, Optional

import numpy as np
import torch
import cv2
import imageio
from torch.fft import fft2, ifft2, fftshift, ifftshift
from tqdm import tqdm

from Wan_SR.pipelines.pipeline_utils import probe_video_meta, open_video_writer

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# =========================
# Args
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse two folders of videos with T+S sliding windows")

    # I/O
    parser.add_argument("--consistent_dir", type=str, required=True, help="Folder of temporally consistent videos")
    parser.add_argument("--sharp_dir", type=str, required=True, help="Folder of sharper videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder")

    # Fusion params
    parser.add_argument("--fc", type=float, default=0.20, help="Butterworth low-pass cutoff frequency")
    parser.add_argument("--alpha", type=float, default=1.0, help="High-frequency fusion strength")
    parser.add_argument("--border", type=int, default=2, help="Hard border width")
    parser.add_argument("--order", type=int, default=2, help="Butterworth filter order")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical epsilon for weight computation")

    # Sliding window
    parser.add_argument("--window_t", type=int, default=33, help="Temporal window size")
    parser.add_argument("--overlap_t", type=int, default=8, help="Temporal overlap")
    parser.add_argument("--window_h", type=int, default=720, help="Spatial window height")
    parser.add_argument("--overlap_h", type=int, default=128, help="Spatial overlap height")
    parser.add_argument("--window_w", type=int, default=1280, help="Spatial window width")
    parser.add_argument("--overlap_w", type=int, default=128, help="Spatial overlap width")

    # Runtime & Meta
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--sort_files", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_quality", type=int, default=9)
    parser.add_argument("--temp_dir", type=str, default=None)
    parser.add_argument("--keep_temp", action="store_true")

    return parser.parse_args()


# =========================
# Basic Sliding Window Utils
# =========================
def compute_window_starts(length: int, window: int, overlap: int) -> List[int]:
    if length <= window: return [0]
    stride = window - overlap
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts

def build_1d_blend(length: int, overlap: int) -> np.ndarray:
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 1: return w
    ov = min(overlap, length // 2)
    ramp = np.linspace(1e-6, 1.0, ov, dtype=np.float32)
    w[:ov] = np.minimum(w[:ov], ramp)
    w[-ov:] = np.minimum(w[-ov:], ramp[::-1])
    return w

def build_2d_blend(h: int, w: int, overlap_h: int, overlap_w: int) -> np.ndarray:
    wy = build_1d_blend(h, overlap_h)
    wx = build_1d_blend(w, overlap_w)
    return (wy[:, None] * wx[None, :]).astype(np.float32)

def linear_blend_overlap(prev_frames: List[np.ndarray], curr_frames: List[np.ndarray], overlap: int) -> List[np.ndarray]:
    if overlap <= 0: return curr_frames
    overlap = min(overlap, len(prev_frames), len(curr_frames))
    prev_tail = np.stack(prev_frames[-overlap:]).astype(np.float32)
    curr_head = np.stack(curr_frames[:overlap]).astype(np.float32)
    alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(overlap, 1, 1, 1)
    blended = (prev_tail * (1.0 - alpha) + curr_head * alpha).clip(0, 255).round().astype(np.uint8)
    return [blended[i] for i in range(overlap)] + curr_frames[overlap:]

# =========================
# Core Fusion Logic (Frequency Domain)
# =========================
def rgb_to_yuv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14713*r - 0.28886*g + 0.436*b
    v = 0.615*r - 0.51499*g - 0.10001*b
    return torch.cat([y, u, v], dim=1)

def yuv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    y, u, v = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    r = y + 1.13983*v
    g = y - 0.39465*u - 0.5806*v
    b = y + 2.03211*u
    return torch.cat([r, g, b], dim=1)

def butterworth_lp(h: int, w: int, fc: float, order: int, device: str) -> torch.Tensor:
    u = torch.linspace(-0.5, 0.5, w, device=device).view(1, 1, 1, w)
    v = torch.linspace(-0.5, 0.5, h, device=device).view(1, 1, h, 1)
    r = torch.sqrt(u**2 + v**2)
    return 1.0 / (1.0 + (r / fc)**(2 * order))

@torch.no_grad()
def fuse_patch(tc: torch.Tensor, ts: torch.Tensor, hlp: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    yuv_c = rgb_to_yuv(tc)
    yuv_s = rgb_to_yuv(ts)
    yc, ys = yuv_c[:, 0:1], yuv_s[:, 0:1]

    def get_hp(y):
        f = fftshift(fft2(y, norm="ortho"))
        ilp = torch.real(ifft2(ifftshift(f * hlp), norm="ortho"))
        return ilp, y - ilp

    lc, hc = get_hp(yc)
    _, hs = get_hp(ys)

    w_hf = hs.abs() / (hs.abs() + hc.abs() + args.eps)
    y_fused = lc + args.alpha * (w_hf * hs + (1.0 - w_hf) * hc)
    
    # Border mask handling
    if args.border > 0:
        h, w = y_fused.shape[-2:]
        mask = torch.zeros_like(y_fused)
        mask[:, :, args.border:h-args.border, args.border:w-args.border] = 1.0
        y_fused = mask * y_fused + (1.0 - mask) * yc

    yuv_out = torch.cat([y_fused, yuv_c[:, 1:]], dim=1)
    return yuv_to_rgb(yuv_out).clamp(0, 1)

# =========================
# Video I/O
# =========================
def read_video_chunk_rgb(path: str, start: int, num: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(num):
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# =========================
# Main Sliding Logic
# =========================
def process_temporal_chunk_with_spatial_windows(
    frames_c: List[np.ndarray],
    frames_s: List[np.ndarray],
    args: argparse.Namespace
) -> List[np.ndarray]:
    t = len(frames_c)
    full_h, full_w = frames_c[0].shape[:2]

    h_starts = compute_window_starts(full_h, args.window_h, args.overlap_h)
    w_starts = compute_window_starts(full_w, args.window_w, args.overlap_w)

    acc = np.zeros((t, full_h, full_w, 3), dtype=np.float32)
    wsum = np.zeros((t, full_h, full_w, 1), dtype=np.float32)

    for top in h_starts:
        ph = min(args.window_h, full_h - top)
        hlp = butterworth_lp(ph, min(args.window_w, full_w), args.fc, args.order, args.device)
        for left in w_starts:
            pw = min(args.window_w, full_w - left)
            if hlp.shape[-1] != pw: # Recompute if edge patch width differs
                 hlp = butterworth_lp(ph, pw, args.fc, args.order, args.device)

            # Extract patches
            pc = [f[top:top+ph, left:left+pw] for f in frames_c]
            ps = [f[top:top+ph, left:left+pw] for f in frames_s]
            
            # Batch process frames in patch
            tc = torch.from_numpy(np.stack(pc)).permute(0, 3, 1, 2).to(args.device).float() / 255.0
            ts = torch.from_numpy(np.stack(ps)).permute(0, 3, 1, 2).to(args.device).float() / 255.0
            
            out_patch = fuse_patch(tc, ts, hlp, args)
            out_patch = (out_patch.permute(0, 2, 3, 1).cpu().numpy() * 255.0)

            # Weighting
            weight = build_2d_blend(ph, pw, args.overlap_h, args.overlap_w)[..., None]
            acc[:, top:top+ph, left:left+pw, :] += out_patch * weight
            wsum[:, top:top+ph, left:left+pw, :] += weight
            
            torch.cuda.empty_cache()

    fused = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype(np.uint8)
    return [fused[i] for i in range(t)]

def process_video_sliding(path_c: str, path_s: str, output_path: str, args: argparse.Namespace):
    in_w, in_h, probed_fps, total_frames = probe_video_meta(path_c)
    fps = args.fps if args.fps and args.fps > 0 else probed_fps
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open_video_writer(output_path, fps=fps, quality=args.save_quality)

    t_starts = compute_window_starts(total_frames, args.window_t, args.overlap_t)
    prev_buffer: Optional[List[np.ndarray]] = None
    prev_start: Optional[int] = None

    for idx, start in enumerate(tqdm(t_starts, desc="Temporal Chunks", leave=False)):
        curr_len = min(args.window_t, total_frames - start)
        fc = read_video_chunk_rgb(path_c, start, curr_len)
        fs = read_video_chunk_rgb(path_s, start, curr_len)

        fused_frames = process_temporal_chunk_with_spatial_windows(fc, fs, args)

        if prev_buffer is None:
            prev_buffer, prev_start = fused_frames, start
            continue

        overlap = (prev_start + len(prev_buffer)) - start
        write_len = max(0, len(prev_buffer) - overlap)
        
        for i in range(write_len):
            writer.append_data(prev_buffer[i])

        prev_buffer = linear_blend_overlap(prev_buffer, fused_frames, max(0, overlap))
        prev_start = start

    if prev_buffer:
        for f in prev_buffer: writer.append_data(f)
    writer.close()

def inference_folder(args: argparse.Namespace):
    video_files = [f for f in os.listdir(args.consistent_dir) if f.lower().endswith(VIDEO_EXTS)]
    if args.sort_files: video_files = sorted(video_files)

    for vname in tqdm(video_files, desc="Processing videos"):
        path_c = os.path.join(args.consistent_dir, vname)
        path_s = os.path.join(args.sharp_dir, vname)
        save_path = os.path.join(args.output_dir, vname)

        if args.skip_existing and os.path.exists(save_path): continue
        if not os.path.exists(path_s): continue

        process_video_sliding(path_c, path_s, save_path, args)

if __name__ == "__main__":
    args = parse_args()
    inference_folder(args)