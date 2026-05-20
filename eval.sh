

merge_type=vision_text_guide  #  concat_text2 film_text_guide text_guide vision_text_guide   

lora_type=merger_save_all_lora-new_label-after-w2c-nf-all_data-new_loss-npy-version_old_trainer-no_norm-${merge_type}-num_3d_freqs-4

lora_type=merger_save_all_lora-new_label-after-w2c-nf-all_data-new_loss-npy-version_2-no_norm-without_3d-gradient_1_gpu_4-with-eval

lora_type=merger_save_all_lora-new_label-after-w2c-nf-all_data-new_loss-npy-version_old_trainer-gradient_1_gpu_4-v3-no_norm-direct_add-num_3d_freqs-4

lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-without_3d

lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-2-no_norm-deep_text_guide-num_3d_freqs-4

lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-text_cross_attention-lambda_sparse-0



lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-text_cross_attention-question_embedding-0-method5-3  # 0.8555  ques: 0.8545  5：0.8545
   # method3: 0.8565  method4:0.8574     method5:0.8565  method5-2：0.8565    method5-3:0.8555
test_data=processed_data_with_depth_npy/test/test_data_multi_view_follow.json
# test_data=processed_data_with_depth_npy/test/test_data_simple_multi_view_follow.json
merge_type=text_cross_attention

# lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-text_cross_scale-question_embedding-0-6  # 0: 0.8584  0.01: 0.8584  question_embedding-0.01:0.8535  q-0：0.8497  q-6:0.8545
# merge_type=text_cross_scale


lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-fusion_add-m53  
merge_type=fusion_add

bash qwen_3d/scripts/eval_3d.sh \
    -t ${test_data} \
    -l ${lora_type} \
    -n no_norm \
    -m ${merge_type} \
    -w True \
    -d nf \
    -f 4 \
    -e True \
    -g 3  \
   #  -o "record"  \
   #  -r True


