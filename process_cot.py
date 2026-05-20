import json

def process_datasets(file_a_path, file_b_path, output_b_path):
    # 1. 读取 JSON 文件 A 和 B
    with open(file_a_path, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
        
    with open(file_b_path, 'r', encoding='utf-8') as f:
        data_b = json.load(f)

    # 2. 构建 File A 的映射字典 (匹配键: "dataset_name/question_id" -> 值: COT)
    cot_mapping = {}
    for item in data_a:
        uav_paths = item.get("uav_paths", {})
        if not uav_paths:
            continue
            
        # 提取第一个 UAV 的路径 (例如 "Real_2_UAVs/Samples/UAV1/23-00000001-UAV1.jpg")
        # first_path = list(uav_paths.values())[0]
        
        # 提取 dataset_name (路径最前面的文件名，以 '/' 分割)
        # dataset_name = first_path.split('/')[0]
        dataset = {
            2: "Real_2_UAVs",
            3: "Sim_3_UAVs",
            5: "Sim5_VQA_UAVs",
            6: "Sim6_VQA_UAVs"
        }
        nums = len(uav_paths)
        dataset_name = dataset[nums]
        question_id = item.get("question_id", "")
        cot_value = item.get("COT", "")
        
        # 组合成与文件 B 中 "video" 完全一致的格式
        match_key = f"{dataset_name}/{question_id}"
        cot_mapping[match_key] = cot_value

    # 3. 遍历并修改 File B 的内容
    target_sentence = "Answer with the option's letter from the given choices directly."
    from tqdm import tqdm
    for item in tqdm(data_b):
        video_name = item.get("video", "")
        
        # 如果能在 A 中找到对应的 COT
        if video_name in cot_mapping:
            matched_cot = cot_mapping[video_name]
            
            # (可选) 如果你需要在外层也加上 COT 字段，可以取消下面这行的注释
            item["COT"] = matched_cot 
            
            if "conversations" in item:
                for conv in item["conversations"]:
                    # 修改 human 的对话：移除特定句子
                    if conv.get("from") == "human":
                        text = conv.get("value", "")
                        
                        # 连同前面的换行符一起替换掉，保持排版整洁
                        if "\n" + target_sentence in text:
                            text = text.replace("\n" + target_sentence, "")
                        else:
                            text = text.replace(target_sentence, "")
                            
                        # 去除首尾可能多余的空格或换行
                        conv["value"] = text.strip()
                        
                    # 修改 gpt 的对话：将选项替换为 COT
                    elif conv.get("from") == "gpt":
                        conv["value"] = matched_cot
        else:
            print(f"未找到对应cot，{video_name}")
            exit()
    # 4. 将修改后的 B 文件数据写入到新的 JSON 文件中
    with open(output_b_path, 'w', encoding='utf-8') as f:
        # indent=2 保证输出的 json 有良好的缩进和可读性
        json.dump(data_b, f, ensure_ascii=False, indent=2)
        
    print(f"处理完成！成功将处理后的数据保存至: {output_b_path}")

import json

def remove_fields_from_json(input_file, output_file):
    try:
        # 1. 加载原始 JSON 数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            print("错误：JSON 文件根节点不是列表格式。")
            return

        # 2. 遍历并删除指定字段
        fields_to_remove = ["model_option", "api_ans"]
        
        for item in data:
            for field in fields_to_remove:
                # 使用 pop(field, None) 安全删除，如果键不存在也不会报错
                item.pop(field, None)

        # 3. 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"✨ 处理完成！已删除字段 {fields_to_remove}")
        print(f"📁 新文件保存至: {output_file}")
        print(f"📊 剩余样本总数: {len(data)}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file}'")
    except Exception as e:
        print(f"运行出错: {e}")


# ================= 使用示例 =================
# 请将 'file_a.json' 和 'file_b.json' 替换为你实际的文件名
# if __name__ == "__main__":
#     FILE_A = "qwen_3d/cot/merged_test_results_with_COT.json" 
#     FILE_B = "processed_data_with_depth_npy/test/test_data_multi_view_follow.json"
#     OUTPUT_FILE = "processed_data_with_depth_npy/test/test_data_multi_view_cot.json"  # 建议输出到新文件，避免覆盖原始数据
    
#     process_datasets(FILE_A, FILE_B, OUTPUT_FILE)


if __name__ == "__main__":
    # 执行转换
    remove_fields_from_json('processed_data_with_depth_npy/test/test_data_multi_view_cot2.json', 'processed_data_with_depth_npy/test/test_data_multi_view_cot2.json')