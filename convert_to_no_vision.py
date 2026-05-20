import json
import re

def remove_uav_image_prefixes(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 编译正则表达式
    # \s* 能够匹配任何空白字符，包括空格和换行符 \n
    pattern = re.compile(r'UAV\d+[:：]\s*<image>\s*')

    for item in data:
        if 'conversations' in item:
            for conv in item['conversations']:
                if conv.get('from') == 'human':
                    original_value = conv['value']
                    
                    # 替换掉所有符合条件的 UAVx:<image>\n 前缀
                    cleaned_value = pattern.sub('', original_value)
                    
                    # 使用 lstrip() 去除可能残留的开头空格或换行，确保文本干净
                    conv['value'] = cleaned_value.lstrip()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("处理完成！问题前缀已全部移除。")

# 运行脚本
input_json = "processed_data_with_depth_npy/train/train_data_multi_view_follow.json"
output_json = "processed_data_with_depth_npy/train/train_data_multi_view_follow_no_vision.json"
remove_uav_image_prefixes(input_json, output_json)