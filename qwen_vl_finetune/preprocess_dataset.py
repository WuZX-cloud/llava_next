import json

def process_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        conversations = item.get("conversations", [])
        options = item.get("options", {})
        correct_answer = item.get("correct_answer", "")

        # 拼接 options 为长字符串：A.xxx\nB.xxx；
        option_str = "\n".join([f"{k}.{v}" for k, v in options.items()])

        for conv in conversations:
            # 处理 human
            if conv.get("from") == "human":
                value = conv.get("value", "")
                # 去掉 <video>\n
                question = value.replace("<video>\n", "")
                conv["value"] = (
                    f"{question}\n"
                    f"{option_str}\n"
                    "Answer with the option's letter from the given choices directly."
                )

            # 处理 gpt
            elif conv.get("from") == "gpt":
                conv["value"] = correct_answer

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_json = "processed_ACB/train/train_data_multi_view2.json"    # 原始文件路径
    output_json = "processed_ACB/train/train_data_multi_view2_follow.json"  # 处理后的文件路径
    process_json(input_json, output_json)
