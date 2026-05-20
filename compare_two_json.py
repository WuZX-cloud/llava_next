import json
import numpy as np

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_matrix(a, b, atol=1e-6, rtol=1e-6):
    """
    比较两个嵌套 list（矩阵），允许浮点误差
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.allclose(a, b, atol=atol, rtol=rtol)

def compare_two_json(json1, json2):
    keys1 = set(json1.keys())
    keys2 = set(json2.keys())
    common_keys = keys1 & keys2

    results = {
        "total_common_keys": len(common_keys),
        "intrinsics_equal": 0,
        "intrinsics_not_equal": 0,
        "extrinsics_equal": 0,
        "extrinsics_not_equal": 0,
        "details": {}
    }

    for key in common_keys:
        intr1 = json1[key].get("intrinsics")
        intr2 = json2[key].get("intrinsics")
        extr1 = json1[key].get("extrinsics")
        extr2 = json2[key].get("extrinsics")

        intr_equal = compare_matrix(intr1, intr2)
        extr_equal = compare_matrix(extr1, extr2)

        results["details"][key] = {
            "intrinsics_equal": intr_equal,
            "extrinsics_equal": extr_equal
        }

        if intr_equal:
            results["intrinsics_equal"] += 1
        else:
            results["intrinsics_not_equal"] += 1

        if extr_equal:
            results["extrinsics_equal"] += 1
        else:
            results["extrinsics_not_equal"] += 1

    return results

if __name__ == "__main__":
    json_path_1 = "processed_data_latest/processed_mapping.json"
    json_path_2 = "qwen_3d/processed_data/processed_mapping.json"

    json1 = load_json(json_path_1)
    json2 = load_json(json_path_2)

    results = compare_two_json(json1, json2)

    print("=== Comparison Summary ===")
    print(f"Common keys: {results['total_common_keys']}")
    print(f"Intrinsics equal: {results['intrinsics_equal']}")
    print(f"Intrinsics not equal: {results['intrinsics_not_equal']}")
    print(f"Extrinsics equal: {results['extrinsics_equal']}")
    print(f"Extrinsics not equal: {results['extrinsics_not_equal']}")
