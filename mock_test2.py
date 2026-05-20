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
def get_mask_robust(input_ids):
        bsz = input_ids.shape[0]
        question_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        
        # 预计算关键字符的 Token ID 集合 (Set)
        # 因为 tokenizer 可能会有多个 ID 对应同一个字符 (比如不同上下文)
        # 这种方法比 encode 字符串要稳健得多
        
        # 1. 找到所有能代表 "\n" 的 ID
        # 这是一个 Trick: 遍历常用词表范围，或者直接硬编码常见 ID
        # 对于 Qwen/Tiktoken，通常换行符比较固定，但我们可以动态检测
        nl_id = processor.tokenizer.encode("\n", add_special_tokens=False)[-1]
        
        # 2. 找到所有能代表 "A" 或 "A." 的 ID
        # 我们不搜 "A." 了，只搜 "A"，然后看它后面是不是 "."
        # 或者更简单：直接搜 "\n" 作为切分点
        
        # === 极简策略：只找最后一个换行符 ===
        # 绝大多数 VQA 格式：Context \n Question \n Options
        # 问题通常被夹在【倒数第二个换行符】和【倒数第一个换行符】之间
        
        for i in range(bsz):
            ids = input_ids[i]
            
            # 找到所有换行符的位置
            # (ids == nl_id).nonzero() 返回所有换行符索引
            nl_indices = (ids == nl_id).nonzero(as_tuple=True)[0]
            
            q_start, q_end = -1, -1
            
            if len(nl_indices) >= 2:
                # 假设结构: ... Context \n Question \n A. ...
                # 倒数第1个 \n : 选项 A 之前
                # 倒数第2个 \n : 问题之前
                q_end = nl_indices[-1]  # 选项前的换行符
                q_start = nl_indices[-2] + 1 # 问题前的换行符之后
                
            elif len(nl_indices) == 1:
                # 假设结构: ... Context \n Question (无选项)
                # 或者: Question \n A. ... (无Context)
                
                # 我们可以结合 "?" 来判断
                q_mark_id = processor.tokenizer.encode("?", add_special_tokens=False)[-1]
                q_mark_indices = (ids == q_mark_id).nonzero(as_tuple=True)[0]
                
                if len(q_mark_indices) > 0:
                    last_q_mark = q_mark_indices[-1]
                    
                    if last_q_mark < nl_indices[0]:
                         # Question? \n Options
                         q_end = last_q_mark + 1
                         q_start = 0 # 从头开始
                    else:
                         # Context \n Question?
                         q_start = nl_indices[0] + 1
                         q_end = last_q_mark + 1
            
            # 如果实在找不到，回退到兜底逻辑 (取最后 N 个 token)
            if q_start == -1 or q_end == -1:
                 valid_len = (ids != processor.tokenizer.pad_token_id).sum()
                 q_end = valid_len - 10 # 避开 <|im_end|>
                 q_start = max(0, q_end - 40) # 假设问题大概 40 个 token
            
            # 赋值
            if q_end > q_start:
                question_mask[i, q_start : q_end] = 1.0
                
        return question_mask
# =======================================================

# 5. 运行并验证
masks = get_mask_robust(input_ids)

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