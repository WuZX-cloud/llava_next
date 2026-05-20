import torch
from transformers import AutoProcessor

# 1. 加载你的 Processor (只需要 Processor，不需要模型)
model_path = "models/Qwen2.5-VL-7B-Instruct" # 或者你的本地路径
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer

# 2. 构造几个典型的 Case
# Case A: 标准选择题
text_a = "UAV1: <image>\nContext info.\nWhich car is red?\nA. The left one\nB. The right one"
# Case B: 没有 Context，直接提问
text_b = "<image>\nIdentify the building.\nA. School\nB. Hospital"
# Case C: 只有问号，没有选项 (非标准情况)
text_c = "<image>\nWhat is this?"

texts = [text_a, text_b, text_c]

# 3. 处理成 Input IDs
inputs = processor(text=texts, return_tensors="pt", padding=True)
input_ids = inputs.input_ids

# 4. 把你的 DataCollator 里的逻辑复制过来 (或者是核心查找逻辑)
# =======================================================
def get_mask(input_ids):
    bsz = input_ids.shape[0]
    question_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    
    # 定义 Token IDs
    tok_newline = tokenizer.encode("\n", add_special_tokens=False)
    tok_opt_a = tokenizer.encode("\nA.", add_special_tokens=False)
    tok_q_mark = tokenizer.encode("?", add_special_tokens=False)
    
    # 辅助函数 (模拟 DataCollator 里的)
    def find_subsequence(haystack, needle):
        n_h, n_n = len(haystack), len(needle)
        for i in range(n_h - n_n + 1):
            if list(haystack[i:i+n_n]) == needle: return i
        return -1

    def find_backwards(haystack, start, needle):
        n_n = len(needle)
        for i in range(start - n_n, -1, -1):
            if list(haystack[i:i+n_n]) == needle: return i
        return -1

    for i in range(bsz):
        ids = input_ids[i].tolist() # 转 list 方便调试，tensor逻辑同理
        
        # 你的逻辑...
        # 1. 找 \nA.
        opt_idx = find_subsequence(ids, tok_opt_a)
        
        q_start, q_end = -1, -1
        
        if opt_idx != -1:
            q_end = opt_idx
            # 向前找 \n
            nl_idx = find_backwards(ids, opt_idx, tok_newline)
            if nl_idx != -1:
                q_start = nl_idx + len(tok_newline)
        else:
            # 找问号
            q_mark_idx = find_backwards(ids, len(ids), tok_q_mark)
            if q_mark_idx != -1:
                q_end = q_mark_idx + len(tok_q_mark)
                nl_idx = find_backwards(ids, q_mark_idx, tok_newline)
                if nl_idx != -1:
                    q_start = nl_idx + len(tok_newline)
        
        if q_start != -1 and q_end != -1:
            question_mask[i, q_start:q_end] = 1.0
            
    return question_mask
# =======================================================

# 5. 运行并验证
masks = get_mask(input_ids)

print("\n=== OFFLINE TEST RESULTS ===")
for i, txt in enumerate(texts):
    mask = masks[i]
    sel_ids = input_ids[i][mask == 1.0]
    result = tokenizer.decode(sel_ids)
    
    print(f"\nOriginal: {txt.replace('<image>', '[IMG]')}") # 简化显示
    print(f"Captured: [{result}]")
    
    # 简单的自动检查
    if i == 0 and "Which car is red?" in result and "Context" not in result:
        print("✅ Case A Passed")
    elif i == 1 and "Identify the building." in result:
        print("✅ Case B Passed")
    elif i == 2 and "What is this?" in result:
        print("✅ Case C Passed")
    else:
        print("❌ FAILED")