import json
from collections import defaultdict
from collections import OrderedDict

def load_task_stats(json_path):
    # task_stats 结构：{task_type: {"correct": x, "total": y}}
    task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # 读取文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 遍历每一条样本
    for item in data:
        task = item["question_type"]
        correct_answer = item["correct_answer"]
        model_answer = item["response"]

        # 统计总数
        task_stats[task]["total"] += 1

        # 判断是否正确
        if model_answer == correct_answer:
            task_stats[task]["correct"] += 1

    return task_stats



def main(json_file):
    task_stats = load_task_stats(json_file)
    output_path = json_file

    order = [
        "Scene Description",
        "Scene Comparison",
        "Observing Posture",

        "Object Recognition",
        "Object Counting",
        "Object Grounding",
        "Object Matching",

        "Quality Assessment",
        "Usability Assessment",
        "Causal Assessment",
        
        "When to Collaborate",
        "What to Collaborate",
        "Who to Collaborate",
        "Why to Collaborate"  
    ]
    total_correct = 0
    total_count = 0

    ordered_stats = OrderedDict()
    for task in order:   # 使用你自定义的顺序
        ordered_stats[task] = task_stats[task]
        stats = task_stats[task]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        ordered_stats[task]['acc'] = acc
        total_correct += stats["correct"] 
        total_count += stats['total']

    ordered_stats["total_acc"] = total_correct/ total_count

    stats_path = output_path.replace(".json", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(ordered_stats, f, indent=2, ensure_ascii=False)

    print("\n========= 结果统计 =========")
    print(f"总准确率: {total_correct / total_count:.4f}\n")
    print("各任务准确率：")
    for task in order:
        stats = task_stats[task]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {task}: {acc:.4f}  ({stats['correct']}/{stats['total']})")


if __name__ == "__main__" :

    import argparse
    
    parser = argparse.ArgumentParser(description="统计正确率")
    parser.add_argument("--json_file", type=str, default="models/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    main(args.json_file)