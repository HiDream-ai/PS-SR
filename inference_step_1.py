from __future__ import annotations

import argparse
import os
import ast
import shutil
import tempfile
from typing import Dict, List, Tuple, Optional

import imageio
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm

from diffsynth import VideoData
from diffsynth.models import load_state_dict

from Wan_SR.pipelines.wan_sr_base import WanVideoSRPipeline_base, ModelConfig
from Wan_SR.pipelines.wan_sr_draft import WanVideoSRPipeline_draft
from Wan_SR.pipelines.pipeline_utils import (
    expand_patch_embedding,
    get_torch_dtype,
    open_video_writer,
    probe_video_meta,
    resolve_wan_model_paths,
)


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# =========================
# Args
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch video SR inference with temporal + spatial sliding windows")

    # I/O
    parser.add_argument("--input_dir", type=str, required=True, help="Input video folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root folder")

    # Model paths
    parser.add_argument("--lora_base_path", type=str, required=True, help="LoRA path for base model")
    parser.add_argument("--lora_draft_path", type=str, required=True, help="LoRA path for draft model")

    parser.add_argument(
        "--wan_model_dir",
        type=str,
        default=None,
        help="Directory containing Wan2.1-T2V-1.3B files. Overrides --dit_path/--umt5_path/--vae_path.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="./dependent_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    )
    parser.add_argument(
        "--umt5_path",
        type=str,
        default="./dependent_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="./dependent_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    )
    parser.add_argument(
        "--ram_path",
        type=str,
        default="./dependent_models/ram_swin_large_14m.pth",
    )
    parser.add_argument(
        "--DAPE_path",
        type=str,
        default="./dependent_models/DAPE.pth",
    )

    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="4K Ultra-clear, Sharp, Fine Details Restored, Temporal Consistency, Natural Colors",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, "
            "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, "
            "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
        ),
    )

    # Inference args
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--denoising_strength", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--timestep_base", type=int, default=699)
    parser.add_argument(
        "--timestep_draft_list",
        type=lambda s: [int(x) for x in ast.literal_eval(s)],
        default=[599, 499, 399],
        help='e.g. --timestep_draft_list "[599,499,399]"',
    )
    parser.add_argument("--k_select", type=float, default=1.5)
    parser.add_argument("--save_quality", type=int, default=9)

    # Meta
    parser.add_argument("--fps", type=float, default=None, help="Override fps")
    parser.add_argument("--sort_files", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    # Sliding window
    parser.add_argument("--window_t", type=int, default=33, help="Temporal window size")
    parser.add_argument("--overlap_t", type=int, default=16, help="Temporal overlap")
    parser.add_argument("--window_h", type=int, default=720, help="Spatial window height")
    parser.add_argument("--overlap_h", type=int, default=128, help="Spatial overlap height")
    parser.add_argument("--window_w", type=int, default=1280, help="Spatial window width")
    parser.add_argument("--overlap_w", type=int, default=128, help="Spatial overlap width")

    # temp
    parser.add_argument("--temp_dir", type=str, default=None, help="Temporary directory")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary patch videos")

    return parser.parse_args()


# =========================
# Basic utils
# =========================
def compute_window_starts(length: int, window: int, overlap: int) -> List[int]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if overlap < 0 or overlap >= window:
        raise ValueError("Require 0 <= overlap < window")

    if length <= window:
        return [0]

    stride = window - overlap
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def build_1d_blend(length: int, overlap: int) -> np.ndarray:
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 1:
        return w

    ov = min(overlap, length // 2)
    if ov > 0:
        ramp = np.linspace(1e-6, 1.0, ov, dtype=np.float32)
        w[:ov] = np.minimum(w[:ov], ramp)
        w[-ov:] = np.minimum(w[-ov:], ramp[::-1])
    return w


def build_2d_blend(h: int, w: int, overlap_h: int, overlap_w: int) -> np.ndarray:
    wy = build_1d_blend(h, overlap_h)
    wx = build_1d_blend(w, overlap_w)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def read_video_chunk_rgb(input_path: str, start_idx: int, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Failed to read frames from {input_path}, start={start_idx}, count={num_frames}")

    return frames


def crop_frames(frames: List[np.ndarray], top: int, left: int, h: int, w: int) -> List[np.ndarray]:
    return [f[top:top + h, left:left + w].copy() for f in frames]


def write_temp_video_rgb(frames: List[np.ndarray], save_path: str, fps: float) -> None:
    writer = imageio.get_writer(save_path, fps=fps, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def to_float_video(frames: List[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)


def to_uint8_video(arr: np.ndarray) -> List[np.ndarray]:
    arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


def linear_blend_overlap(prev_frames: List[np.ndarray], curr_frames: List[np.ndarray], overlap: int) -> List[np.ndarray]:
    if overlap <= 0:
        return curr_frames

    overlap = min(overlap, len(prev_frames), len(curr_frames))
    if overlap <= 0:
        return curr_frames

    prev_tail = to_float_video(prev_frames[-overlap:])
    curr_head = to_float_video(curr_frames[:overlap])

    alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(overlap, 1, 1, 1)
    blended = prev_tail * (1.0 - alpha) + curr_head * alpha
    blended_list = to_uint8_video(blended)

    return blended_list + curr_frames[overlap:]


# =========================
# Model build
# =========================
def build_model_configs(args: argparse.Namespace) -> List[ModelConfig]:
    if args.wan_model_dir is not None:
        return [ModelConfig(path=path) for path in resolve_wan_model_paths(args.wan_model_dir)]

    return [
        ModelConfig(path=args.dit_path),
        ModelConfig(path=args.umt5_path),
        ModelConfig(path=args.vae_path),
    ]


def build_base_pipe(args: argparse.Namespace) -> WanVideoSRPipeline_base:
    tqdm.write("Loading base model...")
    pipe = WanVideoSRPipeline_base.from_pretrained(
        torch_dtype=get_torch_dtype(args.torch_dtype),
        device=args.device,
        model_configs=build_model_configs(args),
        ram_path=args.ram_path,
        DAPE_path=args.DAPE_path,
    )
    pipe.dit.patch_embedding = expand_patch_embedding(pipe.dit.patch_embedding, factor=2)
    pipe.load_lora(pipe.dit, args.lora_base_path, alpha=1)
    pipe.enable_vram_management()
    return pipe


def build_draft_pipe(args: argparse.Namespace) -> WanVideoSRPipeline_draft:
    tqdm.write("Loading draft model...")
    pipe = WanVideoSRPipeline_draft.from_pretrained(
        torch_dtype=get_torch_dtype(args.torch_dtype),
        device=args.device,
        model_configs=build_model_configs(args),
        ram_path=args.ram_path,
        DAPE_path=args.DAPE_path,
    )
    pipe.dit.patch_embedding = expand_patch_embedding(pipe.dit.patch_embedding, factor=2)
    pipe.shave_dit_draft(k_select=args.k_select)

    num_blocks = len(pipe.dit.blocks)
    pipe.dit.fc_layers = nn.ModuleList([nn.Linear(1536 * 2, 1536) for _ in range(num_blocks)])
    pipe.dit.load_state_dict(load_state_dict(args.lora_draft_path))
    pipe.enable_vram_management()
    return pipe


# =========================
# Single patch inference
# =========================
def run_patch_inference(
    pipe_base: WanVideoSRPipeline_base,
    pipe_draft: WanVideoSRPipeline_draft,
    patch_video_path: str,
    patch_w: int,
    patch_h: int,
    num_frames: int,
    args: argparse.Namespace,
):
    video = VideoData(patch_video_path, width=patch_w, height=patch_h)

    video_base, latents_draft_input, latents_feature_list = pipe_base(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_video=video,
        denoising_strength=args.denoising_strength,
        seed=args.seed,
        tiled=False,
        width=patch_w,
        height=patch_h,
        num_frames=num_frames,
        timestep_base=args.timestep_base,
        cfg_scale=args.cfg_scale,
        timestep_draft_list=args.timestep_draft_list,
        k_select=args.k_select,
    )

    video_draft_list = pipe_draft(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_video=video,
        denoising_strength=args.denoising_strength,
        seed=args.seed,
        tiled=False,
        width=patch_w,
        height=patch_h,
        num_frames=num_frames,
        cfg_scale=args.cfg_scale,
        timestep_draft_list=args.timestep_draft_list,
        latents_next=latents_draft_input,
        latents_feature_list=latents_feature_list,
    )

    return video_base, video_draft_list


# =========================
# Spatial fusion on one temporal chunk
# =========================
def process_temporal_chunk_with_spatial_windows(
    pipe_base: WanVideoSRPipeline_base,
    pipe_draft: WanVideoSRPipeline_draft,
    chunk_frames: List[np.ndarray],
    fps: float,
    args: argparse.Namespace,
    work_dir: str,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    t = len(chunk_frames)
    full_h, full_w = chunk_frames[0].shape[:2]

    h_starts = compute_window_starts(full_h, min(args.window_h, full_h), min(args.overlap_h, max(0, min(args.window_h, full_h) - 1)))
    w_starts = compute_window_starts(full_w, min(args.window_w, full_w), min(args.overlap_w, max(0, min(args.window_w, full_w) - 1)))

    num_drafts = len(args.timestep_draft_list)

    acc_base = np.zeros((t, full_h, full_w, 3), dtype=np.float32)
    wsum_base = np.zeros((t, full_h, full_w, 1), dtype=np.float32)

    acc_drafts = [np.zeros((t, full_h, full_w, 3), dtype=np.float32) for _ in range(num_drafts)]
    wsum_drafts = [np.zeros((t, full_h, full_w, 1), dtype=np.float32) for _ in range(num_drafts)]

    patch_idx = 0
    total_patches = len(h_starts) * len(w_starts)

    for top in h_starts:
        patch_h = min(args.window_h, full_h - top)
        for left in w_starts:
            patch_w = min(args.window_w, full_w - left)
            patch_idx += 1
            tqdm.write(f"    Spatial patch {patch_idx}/{total_patches}: top={top}, left={left}, h={patch_h}, w={patch_w}")

            patch_frames = crop_frames(chunk_frames, top, left, patch_h, patch_w)

            patch_video_path = os.path.join(work_dir, f"patch_{top}_{left}.mp4")
            write_temp_video_rgb(patch_frames, patch_video_path, fps=fps)

            video_base, video_draft_list = run_patch_inference(
                pipe_base=pipe_base,
                pipe_draft=pipe_draft,
                patch_video_path=patch_video_path,
                patch_w=patch_w,
                patch_h=patch_h,
                num_frames=t,
                args=args,
            )

            if not args.keep_temp:
                os.remove(patch_video_path)

            weight_2d = build_2d_blend(
                patch_h,
                patch_w,
                overlap_h=min(args.overlap_h, patch_h // 2),
                overlap_w=min(args.overlap_w, patch_w // 2),
            )[..., None]  # [h, w, 1]
            weight_3d = np.broadcast_to(weight_2d[None, ...], (t, patch_h, patch_w, 1))

            base_arr = to_float_video(video_base)
            acc_base[:, top:top + patch_h, left:left + patch_w, :] += base_arr * weight_3d
            wsum_base[:, top:top + patch_h, left:left + patch_w, :] += weight_3d

            for i in range(num_drafts):
                draft_arr = to_float_video(video_draft_list[i])
                acc_drafts[i][:, top:top + patch_h, left:left + patch_w, :] += draft_arr * weight_3d
                wsum_drafts[i][:, top:top + patch_h, left:left + patch_w, :] += weight_3d

            torch.cuda.empty_cache()

    fused_base = acc_base / np.clip(wsum_base, 1e-6, None)
    fused_base_frames = to_uint8_video(fused_base)

    fused_draft_list = []
    for i in range(num_drafts):
        fused_draft = acc_drafts[i] / np.clip(wsum_drafts[i], 1e-6, None)
        fused_draft_list.append(to_uint8_video(fused_draft))

    return fused_base_frames, fused_draft_list


# =========================
# Full video process with temporal sliding
# =========================
def process_video_sliding(
    pipe_base: WanVideoSRPipeline_base,
    pipe_draft: WanVideoSRPipeline_draft,
    input_path: str,
    output_dir: str,
    args: argparse.Namespace,
) -> None:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    tqdm.write(f"Processing video: {base_name}")

    in_w, in_h, probed_fps, total_frames = probe_video_meta(input_path)
    fps = args.fps if args.fps is not None and args.fps > 0 else probed_fps

    tqdm.write(f"  -> input meta: width={in_w}, height={in_h}, fps={fps:.6f}, frames={total_frames}")

    base_out_path = os.path.join(output_dir, "base", f"{base_name}.mp4")
    draft_out_paths = [
        os.path.join(output_dir, f"base+draft_{i+1}", f"{base_name}.mp4")
        for i in range(len(args.timestep_draft_list))
    ]

    if args.skip_existing and os.path.exists(base_out_path):
        tqdm.write(f"  -> skip existing: {base_out_path}")
        return

    os.makedirs(os.path.join(output_dir, "base"), exist_ok=True)
    for i in range(len(args.timestep_draft_list)):
        os.makedirs(os.path.join(output_dir, f"base+draft_{i+1}"), exist_ok=True)

    base_writer = open_video_writer(base_out_path, fps=fps, quality=args.save_quality)
    draft_writers = [
        open_video_writer(draft_out_paths[i], fps=fps, quality=args.save_quality)
        for i in range(len(args.timestep_draft_list))
    ]

    t_window = min(args.window_t, total_frames)
    t_overlap = min(args.overlap_t, max(0, t_window - 1))
    t_starts = compute_window_starts(total_frames, t_window, t_overlap)

    prev_base_buffer: Optional[List[np.ndarray]] = None
    prev_draft_buffers: Optional[List[List[np.ndarray]]] = None
    prev_start: Optional[int] = None

    work_root = args.temp_dir if args.temp_dir is not None else os.path.join(output_dir, "_tmp")
    os.makedirs(work_root, exist_ok=True)

    try:
        for chunk_idx, start in enumerate(tqdm(t_starts, desc=f"Temporal chunks for {base_name}")):
            curr_len = min(t_window, total_frames - start)
            tqdm.write(f"  -> temporal chunk {chunk_idx+1}/{len(t_starts)}: start={start}, len={curr_len}")

            chunk_frames = read_video_chunk_rgb(input_path, start, curr_len)

            chunk_work_dir = os.path.join(work_root, f"{base_name}_chunk_{chunk_idx:04d}")
            os.makedirs(chunk_work_dir, exist_ok=True)

            fused_base_frames, fused_draft_list = process_temporal_chunk_with_spatial_windows(
                pipe_base=pipe_base,
                pipe_draft=pipe_draft,
                chunk_frames=chunk_frames,
                fps=fps,
                args=args,
                work_dir=chunk_work_dir,
            )

            if not args.keep_temp:
                shutil.rmtree(chunk_work_dir, ignore_errors=True)

            if prev_base_buffer is None:
                prev_base_buffer = fused_base_frames
                prev_draft_buffers = fused_draft_list
                prev_start = start
                continue

            assert prev_draft_buffers is not None
            assert prev_start is not None

            overlap = (prev_start + len(prev_base_buffer)) - start
            overlap = max(0, overlap)

            # write prev non-overlap
            write_len = max(0, len(prev_base_buffer) - overlap)
            for i in range(write_len):
                base_writer.append_data(prev_base_buffer[i])
            for di in range(len(draft_writers)):
                for i in range(write_len):
                    draft_writers[di].append_data(prev_draft_buffers[di][i])

            # blend overlap and keep current tail in buffer
            prev_base_buffer = linear_blend_overlap(prev_base_buffer, fused_base_frames, overlap)
            new_draft_buffers = []
            for di in range(len(draft_writers)):
                merged = linear_blend_overlap(prev_draft_buffers[di], fused_draft_list[di], overlap)
                new_draft_buffers.append(merged)
            prev_draft_buffers = new_draft_buffers
            prev_start = start

            torch.cuda.empty_cache()

        # flush last buffer
        if prev_base_buffer is not None:
            for frame in prev_base_buffer:
                base_writer.append_data(frame)
        if prev_draft_buffers is not None:
            for di in range(len(draft_writers)):
                for frame in prev_draft_buffers[di]:
                    draft_writers[di].append_data(frame)

    finally:
        base_writer.close()
        for w in draft_writers:
            w.close()

        if not args.keep_temp and args.temp_dir is None:
            shutil.rmtree(work_root, ignore_errors=True)


# =========================
# Folder inference
# =========================
def inference_folder(args: argparse.Namespace) -> None:
    pipe_base = build_base_pipe(args)
    pipe_draft = build_draft_pipe(args)

    os.makedirs(args.output_dir, exist_ok=True)

    video_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(VIDEO_EXTS)
    ]
    if args.sort_files:
        video_files = sorted(video_files)

    tqdm.write(f"Found {len(video_files)} videos in {args.input_dir}")

    for vname in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(args.input_dir, vname)
        process_video_sliding(
            pipe_base=pipe_base,
            pipe_draft=pipe_draft,
            input_path=input_path,
            output_dir=args.output_dir,
            args=args,
        )


if __name__ == "__main__":
    args = parse_args()
    tqdm.write("Start Testing Videos...")
    inference_folder(args)
