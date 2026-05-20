import json
import re
from pathlib import Path
from collections import Counter, defaultdict


def load_json_dataset(json_path):
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "train", "samples", "annotations", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Unsupported JSON structure. Please check your JSON file.")


def normalize_answer(ans):
    if ans is None:
        return None

    ans = str(ans).strip().upper()

    for letter in ["A", "B", "C", "D"]:
        if ans == letter:
            return letter
        if ans.startswith(letter + ".") or ans.startswith(letter + ":"):
            return letter
        if ans.startswith("OPTION " + letter):
            return letter

    return ans


def normalize_question_type(qtype):
    """
    从 question_type 中提取前面的编号。
    例如：
    '1.1 Scene Description (UAV1)' -> '1.1'
    '4.3 Who to Collaborate (UAV3)' -> '4.3'
    """
    if qtype is None:
        return "Unknown"

    qtype = str(qtype).strip()

    match = re.match(r"^(\d+\.\d+)", qtype)
    if match:
        return match.group(1)

    # 如果没有编号，则保留原始类型，方便排查异常数据
    return qtype


def analyze_by_question_type_prefix(json_path):
    samples = load_json_dataset(json_path)

    type_counter = defaultdict(Counter)
    invalid = []

    for idx, sample in enumerate(samples):
        raw_qtype = sample.get("question_type", "Unknown")
        qtype = normalize_question_type(raw_qtype)

        ans = normalize_answer(sample.get("correct_answer"))

        if ans in ["A", "B", "C", "D"]:
            type_counter[qtype][ans] += 1
        else:
            invalid.append({
                "index": idx,
                "question_type": raw_qtype,
                "correct_answer": sample.get("correct_answer")
            })

    print("=" * 90)
    print("Answer distribution grouped by question_type prefix")
    print("=" * 90)

    for qtype in sorted(type_counter.keys(), key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r"[.]", x)]):
        counter = type_counter[qtype]
        total = sum(counter.values())

        print(f"\nQuestion type: {qtype}")
        print(f"Total: {total}")

        for letter in ["A", "B", "C", "D"]:
            count = counter[letter]
            ratio = count / total * 100 if total > 0 else 0
            print(f"{letter}: {count:6d}  ({ratio:6.2f}%)")

        majority_letter, majority_count = counter.most_common(1)[0]
        majority_acc = majority_count / total * 100 if total > 0 else 0
        print(f"Majority baseline: always choose {majority_letter}, acc = {majority_acc:.2f}%")

    print("\n" + "=" * 90)
    print(f"Invalid / missing correct_answer: {len(invalid)}")

    if invalid:
        print("Examples of invalid answers:")
        for item in invalid[:10]:
            print(item)

    return type_counter, invalid


json_path = "processed_data_with_depth_npy/train/train_data_multi_view_follow_shuffle_options.json"

type_counter, invalid = analyze_by_question_type_prefix(json_path)

