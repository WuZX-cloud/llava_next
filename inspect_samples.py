import json

with open("processed_data_with_depth_npy/test/sim5_sim6_merged_test_data_multi_view_follow.json") as f:
    data = json.load(f)

spatial_types = ["Object Grounding", "Object Matching", "Who to Collaborate", "When to Collaborate"]
for qt in spatial_types:
    print("=" * 60)
    print("Type:", qt)
    print("=" * 60)
    samples = [x for x in data if x.get("question_type") == qt]
    for s in samples[:3]:
        for conv in s["conversations"]:
            if conv["from"] == "human":
                text = conv["value"].replace("<image>", "[IMG]")
                lines = text.split("\n")
                # get lines that are the actual question/options
                q_lines = [l for l in lines if l.strip() and not l.strip().startswith("UAV") and "[IMG]" not in l]
                print("  Q:", " ".join(q_lines[-4:]).strip()[:300])
            elif conv["from"] == "gpt":
                print("  A:", conv["value"][:150])
        print("  video:", s["video"])
        print()
