

checkpoint_dir=output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.31-scene_object-cot-text_cross_attention

output_file=${checkpoint_dir}/adapter_model_recovered.bin

python ${checkpoint_dir}/zero_to_fp32.py ${checkpoint_dir}  ${output_file}

output_file_safe=${checkpoint_dir}/adapter_model.safetensors

python convert_to_safetensors.py ${output_file}  ${output_file_safe}