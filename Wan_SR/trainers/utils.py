import ast
import time

import imageio
import os
import torch
import warnings
import torchvision
import argparse
import json
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
import itertools
from accelerate import Accelerator
import torch.nn as nn

from diffsynth import VideoData
from diffsynth.trainers.utils import DiffusionTrainingModule

from ..models.discriminator_wan import Discriminator
from ..degradation.realesrgan_video import RealESRGAN_video_degradation


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, remove_prefix_in_ckpt_reg=None, state_dict_converter=lambda x: x, save_steps=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.remove_prefix_in_ckpt_reg = remove_prefix_in_ckpt_reg
        self.state_dict_converter = state_dict_converter
        self.save_steps = save_steps

    def on_step_end(self, loss_dict, accelerator, pipe_base=None, pipe_reg=None, pipe_draft=None, discriminator=None, step_id=None, save_latest=True, save=True):
        accelerator.log(loss_dict, step=step_id)
        if self.save_steps is not None and step_id % self.save_steps == 0 and save:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if pipe_base is not None:
                    state_dict = accelerator.get_state_dict(pipe_base)
                    state_dict = accelerator.unwrap_model(pipe_base).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                    state_dict = self.state_dict_converter(state_dict)
                    os.makedirs(self.output_path, exist_ok=True)
                    if save_latest:
                        path = os.path.join(self.output_path, "latest_base.safetensors")
                    else:
                        path = os.path.join(self.output_path, "step-{:07d}_base.safetensors".format(step_id))
                    accelerator.save(state_dict, path, safe_serialization=True)

                if pipe_reg is not None:
                    state_dict = accelerator.get_state_dict(pipe_reg)
                    state_dict = accelerator.unwrap_model(pipe_reg).export_trainable_state_dict(
                        state_dict, remove_prefix=self.remove_prefix_in_ckpt_reg)
                    state_dict = self.state_dict_converter(state_dict)
                    os.makedirs(self.output_path, exist_ok=True)
                    if save_latest:
                        path = os.path.join(self.output_path, "latest_reg.safetensors")
                    else:
                        path = os.path.join(self.output_path, "step-{:07d}_reg.safetensors".format(step_id))
                    accelerator.save(state_dict, path, safe_serialization=True)

                if pipe_draft is not None:
                    state_dict = accelerator.get_state_dict(pipe_draft.module.dit)
                    # state_dict = accelerator.unwrap_model(pipe_draft).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                    state_dict = self.state_dict_converter(state_dict)
                    os.makedirs(self.output_path, exist_ok=True)
                    if save_latest:
                        path = os.path.join(self.output_path, "latest_draft.safetensors")
                    else:
                        path = os.path.join(self.output_path, "step-{:07d}_draft.safetensors".format(step_id))
                    accelerator.save(state_dict, path, safe_serialization=True)

                if discriminator is not None:
                    state_dict = accelerator.get_state_dict(discriminator)
                    os.makedirs(self.output_path, exist_ok=True)
                    if save_latest:
                        path = os.path.join(self.output_path, "latest_discriminator.safetensors")
                    else:
                        path = os.path.join(self.output_path, "step-{:07d}_discriminator.safetensors".format(step_id))
                    accelerator.save(state_dict, path, safe_serialization=True)

        elif (step_id+1) % 50 == 0:
            accelerator.wait_for_everyone()

    def on_epoch_end(self, accelerator, model, epoch_id):
        pass


def launch_training_task_base(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer_base: torch.optim.Optimizer,
    optimizer_reg: torch.optim.Optimizer,
    scheduler_base: torch.optim.lr_scheduler.LRScheduler,
    scheduler_reg: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    args=None,
    layers_to_opt_base=None,
    layers_to_opt_reg=None
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0])
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with="swanlab")
    accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"swanlab": {"experiment_name": args.output_path.split("/")[-1]}})

    discriminator = Discriminator()
    optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)

    # set degradation pipeline
    model.pipe_base.degradation = RealESRGAN_video_degradation(args.deg_file_path, device=accelerator.device)

    pipe_base, pipe_reg, discriminator, optimizer_base, optimizer_reg, optimizer_discriminator, dataloader, scheduler_base, scheduler_reg = accelerator.prepare(
        model.pipe_base, model.pipe_reg, discriminator, optimizer_base, optimizer_reg, optimizer_discriminator, dataloader, scheduler_base, scheduler_reg)

    # load negative prompt
    pipe_base.module.load_negative_context(args.negative_prompt)

    step_id = 0

    def check_nan(tensor):
        return torch.isnan(tensor).any().item()

    for epoch_id in range(num_epochs):
        data_iter = iter(dataloader)
        pbar = tqdm(range(len(dataloader)), desc=f"Epoch {epoch_id}")

        for _ in pbar:
            try:
                data = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"[Step {step_id} | Epoch {epoch_id}] Data loading error: {e}. Skipping.")
                continue

            m_acc = [pipe_base, pipe_reg, discriminator]
            with accelerator.accumulate(*m_acc):
                step_id += 1
                optimizer_base.zero_grad()
                accelerator.print(f"Step {step_id} | Epoch {epoch_id} | Length {len(dataloader)}")

                loss = 0
                loss_dict = {}

                loss_l2, original_difference, inputs = pipe_base.module.training_loss_l2(
                    data, current_timestep=args.timestep_base, next_timestep=args.timestep_draft_list[0])
                loss_dict["original_difference"] = original_difference
                # 1. compute loss_l2
                if args.alpha_l2 > 0:
                    loss += args.alpha_l2 * loss_l2
                    loss_dict["loss_l2"] = loss_l2

                # 2. compute loss_pixel
                if (args.alpha_pixel_l2 > 0 or args.alpha_pixel_lpips > 0) and args.train_video_pixel:
                    loss_pixel_l2, loss_pixel_lpips = pipe_base.module.training_loss_pixel(inputs)
                    loss += args.alpha_pixel_l2 * loss_pixel_l2
                    loss += args.alpha_pixel_lpips * loss_pixel_lpips
                    loss_dict["loss_pixel_l2"] = loss_pixel_l2
                    loss_dict["loss_pixel_lpips"] = loss_pixel_lpips

                models = {name: getattr(pipe_reg.module, name) for name in pipe_reg.module.in_iteration_models}
                # 3. compute loss_reg
                if args.alpha_reg > 0:
                    loss_reg = pipe_reg.module.training_loss_reg(**models, **inputs)
                    loss += args.alpha_reg * loss_reg
                    loss_dict["loss_reg"] = loss_reg

                # 4. compute loss_adv & loss_perc
                if step_id > args.start_adv_step and inputs["LQ_latents"].shape[2] > 1:
                    if args.alpha_adv > 0:
                        discriminator.requires_grad_(False)
                        real_out, real_feat = discriminator(inputs['input_latents'], return_features=True)
                        fake_out, fake_feat = discriminator(inputs['predicted_latents'], return_features=True)  # 不 detach
                        loss_adv = -fake_out.mean()
                        loss += args.alpha_adv * loss_adv
                        loss_dict["loss_adv"] = loss_adv

                    if args.alpha_perc > 0:
                        loss_perc = torch.nn.functional.l1_loss(fake_feat, real_feat.detach(), reduction='mean')
                        loss += args.alpha_perc * loss_perc
                        loss_dict["loss_perc"] = loss_perc

                # 5. update base model
                loss_is_nan = torch.tensor(check_nan(loss), device=accelerator.device, dtype=torch.int)
                if loss_is_nan.item() == 0:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt_base, 1.0)
                    optimizer_base.step()
                    scheduler_base.step()
                else:
                    if accelerator.is_main_process:
                        print("Loss has NaN, skip...")

                inputs['predicted_latents'] = inputs['predicted_latents'].detach()
                inputs['input_latents'] = inputs['input_latents'].detach()
                inputs['context'] = inputs['context'].detach()

                # 6. update discriminator
                if args.alpha_adv > 0 and inputs["LQ_latents"].shape[2] > 1:
                    discriminator.requires_grad_(True)
                    optimizer_discriminator.zero_grad()
                    real_out = discriminator(inputs['input_latents'])   # [B,1,...]
                    fake_out = discriminator(inputs['predicted_latents'])
                    loss_discriminator_real = torch.relu(1.0 - real_out).mean()
                    loss_discriminator_fake = torch.relu(1.0 + fake_out).mean()
                    loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2
                    loss_dict["loss_discriminator"] = loss_discriminator

                    accelerator.backward(loss_discriminator)
                    optimizer_discriminator.step()

                # 7. update reg model
                if args.alpha_reg > 0 and inputs["LQ_latents"].shape[2] > 1:
                    optimizer_reg.zero_grad()
                    loss_diff = pipe_reg.module.training_loss_diff(**models, **inputs)
                    loss_dict["loss_diff"] = loss_diff

                    loss_is_nan = torch.tensor(check_nan(loss_diff), device=accelerator.device, dtype=torch.int)
                    if loss_is_nan.item() == 0:
                        accelerator.backward(loss_diff)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(layers_to_opt_reg, 1.0)
                        optimizer_reg.step()
                        scheduler_reg.step()
                    else:
                        if accelerator.is_main_process:
                            print("Loss_diff has NaN, skip...")

                # 8. log
                model_logger.on_step_end(loss_dict=loss_dict, accelerator=accelerator, pipe_base=pipe_base, pipe_reg=pipe_reg,
                                         discriminator=discriminator, step_id=step_id, save_latest=args.save_latest, save=True)

        model_logger.on_epoch_end(accelerator, model, epoch_id)


def launch_training_task_draft(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer_draft: torch.optim.Optimizer,
    scheduler_draft: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    args=None,
    layers_to_opt_draft=None
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0])
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with="swanlab")
    accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"swanlab": {"experiment_name": args.output_path.split("/")[-1]}})

    # set degradation pipeline
    model.pipe_base.degradation = RealESRGAN_video_degradation(args.deg_file_path, device=accelerator.device)

    pipe_base, pipe_draft, optimizer_draft, dataloader, scheduler_draft = accelerator.prepare(
        model.pipe_base,  model.pipe_draft, optimizer_draft,  dataloader, scheduler_draft)

    # load negative prompt
    pipe_base.module.load_negative_context(args.negative_prompt)

    step_id = 0

    def check_nan(tensor):
        return torch.isnan(tensor).any().item()

    for epoch_id in range(num_epochs):
        data_iter = iter(dataloader)
        pbar = tqdm(range(len(dataloader)), desc=f"Epoch {epoch_id}")

        for _ in pbar:
            try:
                data = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"[Step {step_id} | Epoch {epoch_id}] Data loading error: {e}. Skipping.")
                continue

            m_acc = [pipe_base, pipe_draft]
            with accelerator.accumulate(*m_acc):
                step_id += 1
                optimizer_draft.zero_grad()
                accelerator.print(f"Step {step_id} | Epoch {epoch_id} | Length {len(dataloader)}")

                loss_l2, _, inputs = pipe_base.module.training_loss_l2(
                    data, current_timestep=args.timestep_base, next_timestep=args.timestep_draft_list[0], k_select=args.k_select)

                inputs['predicted_latents'] = inputs['predicted_latents'].detach()
                inputs['input_latents'] = inputs['input_latents'].detach()
                inputs['context'] = inputs['context'].detach()

                for i in range(len(args.timestep_draft_list)):
                    loss = 0
                    loss_dict = {}
                    optimizer_draft.zero_grad()

                    accelerator.print(f"Draft {i+1} | Step {step_id} | Epoch {epoch_id} | Length {len(dataloader)}")

                    if i < len(args.timestep_draft_list)-1:
                        loss_l2, original_difference, inputs = pipe_draft.module.training_loss_l2_draft(
                            inputs, current_timestep=args.timestep_draft_list[i], next_timestep=args.timestep_draft_list[i+1])
                    else:
                        loss_l2, original_difference, inputs = pipe_draft.module.training_loss_l2_draft(
                            inputs, current_timestep=args.timestep_draft_list[i], next_timestep=None)
                    # 1. compute loss_data
                    if args.alpha_l2 > 0:
                        loss += args.alpha_l2 * loss_l2
                        loss_dict[f"loss_l2_draft_{i+1}"] = loss_l2
                        loss_dict[f"original_difference_draft_{i+1}"] = original_difference

                    # 2. compute loss_pixel
                    if (args.alpha_pixel_l2 > 0 or args.alpha_pixel_lpips > 0) and args.train_video_pixel:
                        loss_pixel_l2_draft, loss_pixel_lpips_draft = pipe_draft.module.training_loss_pixel_draft(inputs)
                        loss += args.alpha_pixel_l2 * loss_pixel_l2_draft
                        loss += args.alpha_pixel_lpips * loss_pixel_lpips_draft
                        loss_dict[f"loss_pixel_l2_draft_{i+1}"] = loss_pixel_l2_draft
                        loss_dict[f"loss_pixel_lpips_draft_{i+1}"] = loss_pixel_lpips_draft
                    # 3. update draft model
                    loss_is_nan = torch.tensor(check_nan(loss), device=accelerator.device, dtype=torch.int)
                    if loss_is_nan.item() == 0:
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(layers_to_opt_draft, 1.0)
                        optimizer_draft.step()
                        scheduler_draft.step()
                    else:
                        if accelerator.is_main_process:
                            print("Loss has NaN, skip...")
                    # 4. log
                    model_logger.on_step_end(loss_dict=loss_dict, accelerator=accelerator, pipe_draft=pipe_draft,
                                             step_id=step_id, save_latest=args.save_latest, save=True)
        model_logger.on_epoch_end(accelerator, model, epoch_id)


def wan_sr_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="",
                        required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str,
                        default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720,
                        help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None,
                        help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None,
                        help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video,LQ_video",
                        help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1,
                        help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="Paths to load models. In JSON format.")
    parser.add_argument("--wan_model_dir", type=str, default=None,
                        help="Directory containing Wan2.1-T2V-1.3B model files.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int,
                        default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str,
                        default="./experiments/output", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str,
                        default="dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--remove_prefix_in_ckpt_reg", type=str,
                        default="dit_update.", help="Remove prefix in ckpt in reg.")
    parser.add_argument("--trainable_models", type=str, default=None,
                        help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--trainable_models_reg", type=str, default=None,
                        help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_model_base", type=str,
                        default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_model_reg", type=str,
                        default=None, help="Which model in reg LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int,
                        default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None,
                        help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False,
                        action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint steps.")
    parser.add_argument("--deg_file_path", type=str,
                        default="params_realesrgan.yml", help="Path to the deg file.")
    parser.add_argument("--ram_path", type=str, default="./dependent_models/ram_swin_large_14m.pth",
                        help="Path to the ram file.")

    parser.add_argument("--load_model_base_from", type=str, default=None,
                        help="Path to the model to load.")
    parser.add_argument("--load_model_reg_from", type=str, default=None,
                        help="Path to the model_reg to load.")

    parser.add_argument("--alpha_reg", type=float, default=1.0,
                        help="Alpha for regularization loss.")
    parser.add_argument("--start_adv_step", type=int, default=1000,
                        help="Start adversarial training step.")
    parser.add_argument("--alpha_adv", type=float, default=0.01,
                        help="Alpha for adversarial loss.")
    parser.add_argument("--alpha_perc", type=float, default=0.5,
                        help="Alpha for perceptual loss.")
    parser.add_argument("--alpha_pixel_l2", type=float, default=0,
                        help="Alpha for total variation loss.")
    parser.add_argument("--alpha_pixel_lpips", type=float, default=0,
                        help="Alpha for total variation loss.")
    parser.add_argument("--alpha_l2", type=float, default=1,
                        help="Alpha for total variation loss.")

    parser.add_argument("--project_name", type=str, default="PS-SR",
                        help="Project name for swanlab.")

    parser.add_argument("--timestep_base", type=int, default=699)
    parser.add_argument(
        "--timestep_draft_list",
        type=lambda s: [int(x) for x in ast.literal_eval(s)],
        default=[599, 499, 399],
        help='e.g. --timestep_draft_list "[599,499,399]"',
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="CFG scale for diffusion.")
    parser.add_argument("--negative_prompt", type=str, default="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth",
                        help="Negative prompt for diffusion.")

    parser.add_argument("--save_latest",  type=lambda x: x.lower() == 'true', default=False,
                        help="Save latest checkpoint.")

    parser.add_argument("--train_video_pixel",  type=lambda x: x.lower() == 'true', default=False,
                        help="Train video pixel.")

    parser.add_argument("--k_select", type=float, default=1,
                        help="k value for pipe_draft.dit")

    return parser

def fixed_randint(seed, total_frames, num_frames):
    rng = random.Random(seed)
    return rng.randint(0, total_frames - num_frames)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video"),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def load_video(self, file_path, seed):
        reader = imageio.get_reader(file_path)
        total_frames = int(reader.count_frames())
        num_frames = self.get_num_frames(reader)

        # randomly sample a continuous clip of num_frames frames from the video
        if total_frames > num_frames:
            start_idx = fixed_randint(seed, total_frames, num_frames)
        else:
            start_idx = 0

        frames = []
        for frame_id in range(start_idx, start_idx + num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)

        reader.close()
        return frames

    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension

    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension

    def load_data(self, file_path, seed):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path, seed)
        else:
            return None

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        seed = int(time.time() * 1000) % (2**32)
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path, seed)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
                data['path'] = path
        return data

    def __len__(self):
        return len(self.data) * self.repeat
