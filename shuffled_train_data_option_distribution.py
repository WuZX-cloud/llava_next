import json
import random
import re
from pathlib import Path
from copy import deepcopy


LETTERS = ["A", "B", "C", "D"]


def load_json_dataset(json_path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json_dataset(data, output_path):
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_answer(ans):
    if ans is None:
        return None

    ans = str(ans).strip().upper()

    for letter in LETTERS:
        if ans == letter:
            return letter
        if ans.startswith(letter + ".") or ans.startswith(letter + ":"):
            return letter

    return ans


def split_human_prompt(human_value):
    """
    将 human prompt 拆成：
    1. question_part: UAV1:<image> ... + question
    2. instruction_part: Answer with ...
    
    中间的 A/B/C/D 选项会被重新生成。
    """
    lines = human_value.splitlines()

    first_option_idx = None
    instruction_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 找到第一个选项行，如 A.xxx / A. xxx / A: xxx / A：xxx
        if first_option_idx is None and re.match(r"^A[\.\:\：]", stripped):
            first_option_idx = i

        # 找到最后的指令行
        if "answer with the option" in stripped.lower():
            instruction_idx = i

    if first_option_idx is None:
        raise ValueError("Cannot find option A in human prompt.")

    question_lines = lines[:first_option_idx]

    if instruction_idx is not None:
        instruction_part = lines[instruction_idx].strip()
    else:
        instruction_part = "Answer with the option's letter from the given choices directly."

    question_part = "\n".join(question_lines).rstrip()

    return question_part, instruction_part


def rebuild_human_prompt(question_part, new_options, instruction_part):
    option_lines = []
    for letter in LETTERS:
        option_lines.append(f"{letter}.{new_options[letter]}")

    return (
        question_part
        + "\n"
        + "\n".join(option_lines)
        + "\n"
        + instruction_part
    )


def shuffle_one_sample(sample, rng=random):
    sample = deepcopy(sample)

    if "options" not in sample:
        raise ValueError("Sample does not contain 'options' field.")

    old_options = sample["options"]
    old_correct = normalize_answer(sample.get("correct_answer"))

    if old_correct not in LETTERS:
        raise ValueError(f"Invalid correct_answer: {sample.get('correct_answer')}")

    old_correct_text = old_options[old_correct]

    # 取出原来的选项文本并打乱
    option_texts = [old_options[letter] for letter in LETTERS]
    rng.shuffle(option_texts)

    # 生成新的 options
    new_options = {
        letter: text
        for letter, text in zip(LETTERS, option_texts)
    }

    # 找到正确答案文本现在对应的新字母
    new_correct = None
    for letter, text in new_options.items():
        if text == old_correct_text:
            new_correct = letter
            break

    if new_correct is None:
        raise ValueError("Cannot find new correct answer after shuffling.")

    # 重建 human prompt
    conversations = sample["conversations"]
    human_value = conversations[0]["value"]

    question_part, instruction_part = split_human_prompt(human_value)
    new_human_value = rebuild_human_prompt(
        question_part=question_part,
        new_options=new_options,
        instruction_part=instruction_part
    )

    sample["options"] = new_options
    sample["correct_answer"] = new_correct
    sample["conversations"][0]["value"] = new_human_value
    sample["conversations"][1]["value"] = new_correct

    return sample


def shuffle_dataset_options(input_json_path, output_json_path, seed=42):
    rng = random.Random(seed)

    data = load_json_dataset(input_json_path)

    # 兼容最外层是 list 的情况
    if isinstance(data, list):
        shuffled_data = []
        failed = []

        for idx, sample in enumerate(data):
            try:
                shuffled_data.append(shuffle_one_sample(sample, rng))
            except Exception as e:
                failed.append((idx, str(e)))
                shuffled_data.append(sample)

        save_json_dataset(shuffled_data, output_json_path)

    # 兼容最外层是 dict，数据在 data/train/samples/items 等字段中的情况
    elif isinstance(data, dict):
        shuffled_data = deepcopy(data)
        target_key = None

        for key in ["data", "train", "samples", "annotations", "items"]:
            if key in data and isinstance(data[key], list):
                target_key = key
                break

        if target_key is None:
            raise ValueError("Cannot find list-like dataset field in JSON dict.")

        failed = []
        new_samples = []

        for idx, sample in enumerate(data[target_key]):
            try:
                new_samples.append(shuffle_one_sample(sample, rng))
            except Exception as e:
                failed.append((idx, str(e)))
                new_samples.append(sample)

        shuffled_data[target_key] = new_samples
        save_json_dataset(shuffled_data, output_json_path)

    else:
        raise ValueError("Unsupported JSON structure.")

    print(f"Saved shuffled dataset to: {output_json_path}")
    print(f"Failed samples: {len(failed)}")

    if failed:
        print("First 10 failed samples:")
        for item in failed[:10]:
            print(item)


# 使用方式
input_json_path = "processed_data_with_depth_npy/train/train_data_multi_view_follow.json"
output_json_path = "processed_data_with_depth_npy/train/train_data_multi_view_follow_shuffle_options.json"

shuffle_dataset_options(
    input_json_path=input_json_path,
    output_json_path=output_json_path,
    seed=42
)