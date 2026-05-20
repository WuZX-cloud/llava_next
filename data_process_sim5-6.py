import json

# 输入文件
# input_path = "processed_ACB/test/test_data_multi_view_follow.json"
input_path = "processed_ACB/train/train_data_multi_view_follow.json"


# 输出文件
sim6_path = "processed_ACB/train/sim6_train_data_multi_view_follow.json"
sim5_path = "processed_ACB/train/sim5_train_data_multi_view_follow.json"
merged_path = "processed_ACB/train/sim5_sim6_merged_train_data_multi_view_follow.json"

# 读取原始 JSON（列表）
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 分别筛选
sim6_data = []
sim5_data = []

for item in data:
    dataset = item.get("dataset", "")
    if "Sim_6_UAVs" in dataset:
        sim6_data.append(item)
    if "Sim_5_UAVs" in dataset:
        sim5_data.append(item)

# 合并（如果同一个元素不会同时属于 Sim5 和 Sim6，这样即可）
merged_data = sim6_data + sim5_data

# 保存结果
with open(sim6_path, "w", encoding="utf-8") as f:
    json.dump(sim6_data, f, ensure_ascii=False, indent=2)

with open(sim5_path, "w", encoding="utf-8") as f:
    json.dump(sim5_data, f, ensure_ascii=False, indent=2)

with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"Sim6_VQA 条目数: {len(sim6_data)}")
print(f"Sim5_VQA 条目数: {len(sim5_data)}")
print(f"合并后条目数: {len(merged_data)}")




"""
root@d56dff88e85b:/workspace# python data_process_sim5-6.py
test:
Sim6_VQA 条目数: 135
Sim5_VQA 条目数: 147
合并后条目数: 282
root@d56dff88e85b:/workspace# python data_process_sim5-6.py
train:
Sim6_VQA 条目数: 1848
Sim5_VQA 条目数: 1009
合并后条目数: 2857
"""