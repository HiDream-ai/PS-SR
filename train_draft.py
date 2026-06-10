import torch
import os
import json
from diffsynth.trainers.utils import DiffusionTrainingModule

import torch
import torch.nn as nn

from Wan_SR.pipelines.wan_sr_base import WanVideoSRPipeline_base, ModelConfig
from Wan_SR.pipelines.wan_sr_reg import WanVideoSRPipeline_reg
from Wan_SR.pipelines.wan_sr_draft import WanVideoSRPipeline_draft
from Wan_SR.pipelines.pipeline_utils import expand_patch_embedding, resolve_wan_model_paths
from Wan_SR.trainers.utils import ModelLogger, VideoDataset, wan_sr_parser, launch_training_task_draft

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanSR_Base_TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        wan_model_dir=None,
        model_id_with_origin_paths=None,
        trainable_models_base=None,
        trainable_models_reg=None,
        lora_model_base=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        ram_path=None,
        load_model_base_from=None,
        cfg_scale=None,
        timestep=None,
        is_pervious_lora_version=None,
        k_select=None,
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
        self.pipe_draft = WanVideoSRPipeline_draft.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, ram_path=ram_path)

        # Reset training scheduler
        self.pipe_base.scheduler.set_timesteps(1000, training=True)
        self.pipe_reg.scheduler.set_timesteps(1000, training=True)
        self.pipe_draft.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        self.pipe_base.freeze_except([] if trainable_models_base is None else trainable_models_base.split(","))
        self.pipe_reg.freeze_except([] if trainable_models_reg is None else trainable_models_reg.split(","))
        self.pipe_draft.freeze_except(["dit"])

        # Add LoRA to the base models
        if lora_model_base is not None:
            model = self.add_lora_to_model(getattr(self.pipe_base, lora_model_base),
                                           target_modules=lora_target_modules.split(","), lora_rank=lora_rank)
            setattr(self.pipe_base, lora_model_base, model)
            # Allow patch embedding training
            self.pipe_base.allow_dit_patch_embedding_train()
            # Allow head training
            self.pipe_base.allow_dit_head_train()
            # Resume LoRA training
            if load_model_base_from is not None and is_pervious_lora_version:
                print("Loading LoRA from", load_model_base_from)
                self.pipe_base.load_lora_trainable(self.pipe_base.dit, load_model_base_from, alpha=1)

            self.pipe_base.dit.patch_embedding = expand_patch_embedding(self.pipe_base.dit.patch_embedding, factor=2)

            if load_model_base_from is not None and not is_pervious_lora_version:
                print("Loading LoRA from", load_model_base_from)
                self.pipe_base.load_lora_trainable(self.pipe_base.dit, load_model_base_from, alpha=1)

        # Construct Draft Model
        if is_pervious_lora_version:
            self.pipe_draft.load_lora(self.pipe_draft.dit, load_model_base_from, alpha=1)
            self.pipe_draft.dit.patch_embedding = expand_patch_embedding(self.pipe_draft.dit.patch_embedding, factor=2)
        else:
            self.pipe_draft.dit.patch_embedding = expand_patch_embedding(self.pipe_draft.dit.patch_embedding, factor=2)
            self.pipe_draft.load_lora(self.pipe_draft.dit, load_model_base_from, alpha=1)

        self.pipe_draft.shave_dit_draft(k_select=k_select)  # Select blocks for the draft model

        # Construct FC layers for feature fusion
        dim = 1536
        num_blocks = len(self.pipe_draft.dit.blocks)
        self.pipe_draft.dit.fc_layers = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(num_blocks)])

        # Initialize the FC layers to perform identity mapping at the beginning (only using the features from the draft model)
        with torch.no_grad():
            for fc_layer in self.pipe_draft.dit.fc_layers:
                fc_layer.weight.zero_()
                fc_layer.bias.zero_()
                fc_layer.weight[:, :dim] = torch.eye(dim)  # Make sure the FC layer is identity mapping at the beginning

        # Set dtype and training status for the FC layers
        for fc_layer in self.pipe_draft.dit.fc_layers:
            fc_layer.to(dtype=torch.bfloat16)
            fc_layer.train()


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
        trainable_models_base=args.trainable_models,
        lora_model_base=args.lora_model_base,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        ram_path=args.ram_path,
        load_model_base_from=args.load_model_base_from,
        cfg_scale=args.cfg_scale,
        k_select=args.k_select,
    )


    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        remove_prefix_in_ckpt_reg=args.remove_prefix_in_ckpt_reg,
        save_steps=args.save_steps,
    )

    layers_to_opt_draft = model.pipe_draft.trainable_modules()
    optimizer_draft = torch.optim.AdamW(layers_to_opt_draft, lr=args.learning_rate)
    scheduler_draft = torch.optim.lr_scheduler.ConstantLR(optimizer_draft)

    launch_training_task_draft(
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        optimizer_draft=optimizer_draft,
        scheduler_draft=scheduler_draft,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        args=args,
        layers_to_opt_draft=layers_to_opt_draft,
    )
