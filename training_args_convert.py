import json
import torch

path = "output_3d/merger_save_all_lora-new_label-after-w2c-nf-all_data-new_loss-npy-version_2-no_norm-without_3d-gradient_1_gpu_4/training_args.bin"

output_path = path.replace(".bin", ".json")

training_args = torch.load(path)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(training_args.to_dict(), f, indent=2, ensure_ascii=False)