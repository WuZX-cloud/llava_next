import json
import re

def clean_uav_prefixes(input_file, output_file):
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历列表中的每一个样本
    for item in data:
        # 确保样本中包含 conversations 字段
        if 'conversations' in item:
            for conv in item['conversations']:
                # 只处理 from 为 'human' 的对话
                if conv.get('from') == 'human':
                    # 使用正则表达式替换 UAV加上数字和冒号
                    # \d+ 代表 1 个或多个数字（兼容 UAV1 到 UAV6 甚至更多）
                    # [:：] 兼容英文冒号 ":" 和中文冒号 "："
                    original_value = conv['value']
                    cleaned_value = re.sub(r'UAV\d+[:：]', '', original_value)
                    conv['value'] = cleaned_value

    # 将处理后的数据写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证中文正常显示，indent=2 保持 JSON 格式美观
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("处理完成！结果已保存至", output_file)

# 运行脚本（请将 'input.json' 替换为你的实际文件名）
clean_uav_prefixes('processed_data_with_depth_npy/test/test_data_multi_view_follow.json', 'processed_data_with_depth_npy/test/test_data_multi_view_follow_no_uavid.json')