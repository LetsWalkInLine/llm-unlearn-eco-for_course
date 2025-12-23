import sys
import os
import json
import torch
import numpy as np
from transformers import GenerationConfig

# Add parent directory to path to import eco
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eco.model import HFModel
from eco.optimizer import ZerothOrderOptimizerScalar
from eco.attack.utils import apply_corruption_hook, get_nested_attr, remove_hooks

# --- 1. 设置 ---

MODEL_NAME = "Qwen1.5-4B-Chat"
SENSITIVE_KEYWORDS = ["Harry", "Potter"]
TARGET_ANSWER = "Harry" # 我们想要抑制的答案的第一个 Token
USER_QUERY = "Who is Harry Potter?"

print(f"Loading model: {MODEL_NAME}...")
model = HFModel(
    model_name=MODEL_NAME, 
    config_path="../config/model_config",
    generation_config=GenerationConfig(
        do_sample=False, max_new_tokens=256, use_cache=True
    )
)

tokenizer = model.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. 准备数据和 Mask ---

# 应用聊天模板
prompt_str = tokenizer.apply_chat_template(
    [{"role": "user", "content": USER_QUERY}],
    tokenize=False,
    add_generation_prompt=True,
)

# 分词
inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
input_ids = inputs.input_ids
prompt_len = input_ids.shape[1]

# 创建腐蚀 Mask (pos)
# 我们手动查找包含敏感关键词的 Token 索引
mask = [0] * prompt_len
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print("\n--- Token 分析 ---")
token_analysis = []
for i, token in enumerate(tokens):
    # 解码单个 Token 以处理 BPE 伪影 (如 Ġ)
    decoded_token = tokenizer.decode([input_ids[0][i]])
    is_sensitive = any(kw in decoded_token for kw in SENSITIVE_KEYWORDS)
    if is_sensitive:
        mask[i] = 1
        print(f"Token {i}: '{decoded_token}' -> MASKED")
    else:
        print(f"Token {i}: '{decoded_token}'")
    
    token_analysis.append({
        "index": i,
        "token": decoded_token,
        "masked": bool(mask[i])
    })

# --- 调试：强制全覆盖 Mask ---
# 用户希望看到全 Mask 生效。我们将覆盖上面的逻辑。
# mask = [1] * prompt_len
# print(f"DEBUG: 已强制将 Mask 设置为全 1 (全覆盖模式)。")

# 重要提示：Prompt 包含系统提示和用户查询。
# 但模型生成的是 ANSWER。
# 腐蚀应该应用于 PROMPT Token，以便在模型开始生成答案之前破坏其内部状态。
# 然而，Qwen 的聊天模板可能会将 "Harry Potter" 部分放在后面。
# 让我们仔细检查是否 Mask 了正确的内容。

# 此外，对于 Qwen/GPT 模型，有时 "Harry Potter" 中的 "Harry" 会被拆分或带有前缀。
# 让我们确保捕获到了它们。

print(f"生成的 Mask: {mask}")
if sum(mask) == 0:
    print("警告: 没有 Token 被 Mask！优化将失败。")

# 用于概率计算的目标 ID
target_ids = tokenizer(TARGET_ANSWER, return_tensors="pt").input_ids.to(model.device)
target_id = target_ids[0, 0] # "Harry" 的第一个 Token

# --- 3. 优化循环 ---

def objective_function(strength, model, input_ids, target_id, mask):
    # 1. 应用腐蚀 Hook
    # 我们直接使用底层 API，类似于 demo.py
    # 关键修复：'pos' 期望一个列表的列表 (batch_size, seq_len)
    # 并且它必须与 input_ids 的长度完全匹配。
    
    # 调试：打印攻击模块以确保我们 Hook 到了正确的东西
    # print(f"Attacking module: {model.model_config['attack_module']}")
    
    hook = apply_corruption_hook(
        get_nested_attr(model.model, model.model_config["attack_module"]),
        corrupt_method="rand_noise_first_n",
        corrupt_args={
            "pos": [mask], # 每个 batch item 一个列表
            "dims": 8,    # 维度设为 8，比 32 更难，需要更大的 Strength 才能生效，有助于拉长曲线
            "strength": strength
        },
    )
    
    # 2. 前向传播获取概率
    # 我们必须确保 Hook 实际上被触发了。
    # Hook 注册在 Embedding 层上。
    # 当我们调用 model.model(input_ids) 时，Embedding 层会被调用。
    
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids)
        # 输出 logits 对应于每个位置的下一个 Token 的预测。
        # 我们想要 Prompt 结束后的那个 Token 的预测。
        # 所以我们看序列中最后一个 Token 的 logits。
        logits = outputs.logits[0, -1, :] 
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[target_id].item()
        
    # 3. 移除 Hook (为下一步清理)
    remove_hooks(model.model)
    
    return target_prob

def generate_text(model, input_ids, mask, strength):
    # 应用 Hook
    
    apply_corruption_hook(
        get_nested_attr(model.model, model.model_config["attack_module"]),
        corrupt_method="rand_noise_first_n",
        corrupt_args={
            "pos": [mask],
            "dims": 8,
            "strength": strength
        },
    )
    
    # 生成
    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=input_ids, 
            max_new_tokens=20, 
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 移除 Hook
    remove_hooks(model.model)
    
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

logs = []

print("\n--- 开始优化 ---")

# 优化器设置
# 调整：为了获得平滑的下降曲线，我们大幅降低学习率，并使用较小的维度
lr = 8  # 降低 LR，让它慢慢走
initial_strength = 0.1 # 从很小的噪声开始
eps = 0.05 
beta = initial_strength
min_beta = 0.001
num_steps = 50

optimizer = ZerothOrderOptimizerScalar(
    lr=lr, eps=eps, beta=beta, min_beta=min_beta
)

# 初始状态 (无腐蚀)
initial_prob = objective_function(0.0, model, input_ids, target_id, mask)
initial_text = generate_text(model, input_ids, mask, 0.0)
print(f"初始状态: Prob={initial_prob:.4f}, Output='{initial_text}'")

logs.append({
    "step": -1,
    "strength": 0.0,
    "prob": initial_prob,
    "generated": initial_text
})

for i in range(num_steps):
    # 步进优化
    # 注意：optimizer.step 内部会用 (beta+eps) 和 (beta-eps) 调用 objective_function
    output = optimizer.step(
        objective_function, 
        {
            "model": model, 
            "input_ids": input_ids, 
            "target_id": target_id, 
            "mask": mask
        }
    )
    
    current_strength = optimizer.beta
    current_prob = output["f_score"] # 这是在 (beta + eps) 处的概率
    
    # 生成文本用于记录 (使用当前 beta)
    current_text = generate_text(model, input_ids, mask, current_strength)
    
    print(f"步骤 {i}: Strength={current_strength:.4f}, Prob={current_prob:.4f}")
    
    logs.append({
        "step": i,
        "strength": current_strength,
        "prob": current_prob,
        "generated": current_text
    })
    
    if current_prob < 0.001: # 早停
        print("目标概率足够低。停止。")
        break

# 保存日志
log_path = os.path.join(os.path.dirname(__file__), "demo_logs.json")
with open(log_path, "w") as f:
    json.dump(logs, f, indent=2)

print(f"日志已保存至 {log_path}")

# 保存元数据 (Token 分析和 Mask)
metadata = {
    "tokens": token_analysis,
    "mask": mask
}
metadata_path = os.path.join(os.path.dirname(__file__), "demo_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"元数据已保存至 {metadata_path}")
