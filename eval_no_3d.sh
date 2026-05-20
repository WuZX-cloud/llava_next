

merge_type=vision_text_guide  #  concat_text2 film_text_guide text_guide vision_text_guide   

lora_type=merger_save_all_lora-new_label-after-w2c-nf-all_data-new_loss-npy-version_old_trainer-no_norm-${merge_type}-num_3d_freqs-4

lora_type=merger_save_all_lora-new_label-after-w2c-nf-all_data-new_loss-npy-version_2-no_norm-without_3d-gradient_1_gpu_4-with-eval

lora_type=merger_save_all_lora-new_label-after-w2c-nf-all_data-new_loss-npy-version_old_trainer-gradient_1_gpu_4-v3-no_norm-direct_add-num_3d_freqs-4

lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-without_3d

lora_type=merger_save_all_lora-without_3d_gradient_1_gpu_4-v4

lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-no_3d-m53

bash qwen_3d/scripts/eval_3d.sh \
    -t processed_data_with_depth_npy/test/test_data_multi_view_follow.json \
    -l ${lora_type} \
    -n no_norm \
    -m concat_text \
    -w True \
    -d nf \
    -f 4 \
    -e False \
    -g 7


