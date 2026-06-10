python inference_step_1.py \
    --input_dir videos_for_test/input \
    --output_dir videos_for_test/output \
    --lora_base_path ./checkpoints/pretrained_models/base.safetensors \
    --lora_draft_path ./checkpoints/pretrained_models/draft.safetensors \
    --wan_model_dir ./dependent_models/Wan2.1-T2V-1.3B
