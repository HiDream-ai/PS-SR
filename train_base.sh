# sinlge-GPU training
accelerate launch \
--config_file ./config/accelerate_config_single.yaml \
 train_base.py \
--dataset_base_path ..datasets/YouHQ \
--dataset_metadata_path ./metadata_YouHQ.csv \
--dataset_repeat 10 \
--wan_model_dir ./dependent_models/Wan2.1-T2V-1.3B \
--learning_rate 5e-5 \
--num_epochs 20 \
--output_path ./experiments/train_base \
--lora_model_base dit \
--lora_model_reg dit_update \
--lora_target_modules q,k,v,o,ffn.0,ffn.2 \
--lora_rank 32 \
--save_steps 200 \
--num_frames 33 \
--height 720 \
--width 1280 \
--start_adv_step 500 \
--alpha_adv 0.0 \
--alpha_perc 0.0 \
--alpha_pixel_l2 1.0 \
--alpha_pixel_lpips 2.0 \
--cfg_scale 5.0 \
--train_video_pixel True \
--save_latest True

# multi-GPU training
accelerate launch \
--config_file ./config/accelerate_config_multi.yaml \
--machine_rank ${RANK} \
--main_process_ip ${MASTER_ADDR} \
--main_process_port ${MASTER_PORT} \
 train_base.py \
--dataset_base_path ..datasets/YouHQ \
--dataset_metadata_path ./metadata_YouHQ.csv \
--dataset_repeat 10 \
--wan_model_dir ./dependent_models/Wan2.1-T2V-1.3B \
--learning_rate 5e-5 \
--num_epochs 20 \
--output_path ./experiments/train_base \
--lora_model_base dit \
--lora_model_reg dit_update \
--lora_target_modules q,k,v,o,ffn.0,ffn.2 \
--lora_rank 32 \
--save_steps 200 \
--num_frames 33 \
--height 720 \
--width 1280 \
--start_adv_step 500 \
--alpha_adv 0.0 \
--alpha_perc 0.0 \
--alpha_pixel_l2 1.0 \
--alpha_pixel_lpips 2.0 \
--cfg_scale 5.0 \
--train_video_pixel True \
--save_latest True