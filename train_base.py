import torch
import os
import json
from diffsynth.trainers.utils import DiffusionTrainingModule

import torch
import torch.nn as nn

from Wan_SR.pipelines.wan_sr_base import WanVideoSRPipeline_base, ModelConfig
from Wan_SR.pipelines.wan_sr_reg import WanVideoSRPipeline_reg
from Wan_SR.pipelines.pipeline_utils import expand_patch_embedding, resolve_wan_model_paths
from Wan_SR.trainers.utils import ModelLogger, VideoDataset, wan_sr_parser, launch_training_task_base


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanSR_Base_TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        wan_model_dir=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        trainable_models_reg=None,
        lora_model_base=None,
        lora_model_reg=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        ram_path=None,
        load_model_base_from=None,
        load_model_reg_from=None,
        cfg_scale=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        elif wan_model_dir is not None:
            model_configs += [ModelConfig(path=path) for path in resolve_wan_model_paths(wan_model_dir)]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        if not model_configs:
            raise ValueError("Please provide --wan_model_dir, --model_paths, or --model_id_with_origin_paths.")
        self.pipe_base = WanVideoSRPipeline_base.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, ram_path=ram_path)
        self.pipe_reg = WanVideoSRPipeline_reg.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, cfg_scale=cfg_scale)

        # Reset training scheduler
        self.pipe_base.scheduler.set_timesteps(1000, training=True)
        self.pipe_reg.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        self.pipe_base.freeze_except([] if trainable_models is None else trainable_models.split(","))
        self.pipe_reg.freeze_except([] if trainable_models_reg is None else trainable_models_reg.split(","))

        # Add LoRA to the base models
        if lora_model_base is not None:
            model = self.add_lora_to_model(getattr(self.pipe_base, lora_model_base),
                                           target_modules=lora_target_modules.split(","), lora_rank=lora_rank)
            setattr(self.pipe_base, lora_model_base, model)
            # Allow patch embedding training
            self.pipe_base.allow_dit_patch_embedding_train()
            # Allow head training
            self.pipe_base.allow_dit_head_train()
            # Expand patch embedding
            self.pipe_base.dit.patch_embedding = expand_patch_embedding(self.pipe_base.dit.patch_embedding, factor=2)
            # Resume LoRA training
            if load_model_base_from is not None:
                print("Loading LoRA from", load_model_base_from)
                self.pipe_base.load_lora_trainable(self.pipe_base.dit, load_model_base_from, alpha=1)

        # Add LoRA to the base models reg
        if lora_model_reg is not None:
            model = self.add_lora_to_model(getattr(self.pipe_reg, lora_model_reg),
                                           target_modules=lora_target_modules.split(","), lora_rank=lora_rank)
            setattr(self.pipe_reg, lora_model_reg, model)
            # Allow patch embedding training
            self.pipe_reg.allow_dit_patch_embedding_train()
            # Allow head training
            self.pipe_reg.allow_dit_head_train()
            # Resume LoRA training
            if load_model_reg_from is not None:
                print("Loading LoRA reg from", load_model_reg_from)
                self.pipe_reg.load_lora_trainable(self.pipe_reg.dit_update, load_model_reg_from, alpha=1)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []


if __name__ == "__main__":
    parser = wan_sr_parser()
    args = parser.parse_args()

    dataset = VideoDataset(args=args)
    model = WanSR_Base_TrainingModule(
        model_paths=args.model_paths,
        wan_model_dir=args.wan_model_dir,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        trainable_models_reg=args.trainable_models_reg,
        lora_model_base=args.lora_model_base,
        lora_model_reg=args.lora_model_reg,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        ram_path=args.ram_path,
        load_model_base_from=args.load_model_base_from,
        load_model_reg_from=args.load_model_reg_from,
        cfg_scale=args.cfg_scale,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        remove_prefix_in_ckpt_reg=args.remove_prefix_in_ckpt_reg,
        save_steps=args.save_steps,
    )

    layers_to_opt_base = model.pipe_base.trainable_modules()
    optimizer = torch.optim.AdamW(layers_to_opt_base, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    layers_to_opt_reg = model.pipe_reg.trainable_modules()
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate)
    scheduler_reg = torch.optim.lr_scheduler.ConstantLR(optimizer_reg)

    launch_training_task_base(
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        optimizer_base=optimizer,
        optimizer_reg=optimizer_reg,
        scheduler_base=scheduler,
        scheduler_reg=scheduler_reg,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        args=args,
        layers_to_opt_base=layers_to_opt_base,
        layers_to_opt_reg=layers_to_opt_reg
    )
