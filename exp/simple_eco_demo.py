import sys
import os
import json
import torch
import numpy as np
from transformers import GenerationConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Add parent directory to path to import eco
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eco.model import HFModel
from eco.optimizer import ZerothOrderOptimizerScalar
from eco.attack.utils import apply_corruption_hook, get_nested_attr, remove_hooks

# --- 1. 设置 ---

MODEL_NAME = "Qwen1.5-4B-Chat"
SENSITIVE_KEYWORDS = ["Harry", "Potter"]
TARGET_ANSWER = "Harry" # 本实践想要抑制的答案的第一个 Token
USER_QUERY = "Who is Harry Potter?"

# --- 1.5 模式识别模块 (Pattern Recognition) ---
# 本实践引入一个基于 SVM 的分类器作为“安全门控”。
# 数据集：构建自 SQuAD 和 Wikipedia 的 "HP-Sensitivity" 子集。

def train_gatekeeper():
    print("Training Pattern Recognizer (SVM Classifier)...")
    
    # --- 1. 正样本 (Sensitive): 特定领域的敏感问题 ---
    # 在实际场景中，这通常来自特定任务的 "Forget Set" (如 TOFU 数据集)
    # 这里本实践定义关于 "Harry Potter" 的领域知识为敏感数据
    positive_samples = [
        "Who is Harry Potter?",
        "What house is Harry in at Hogwarts?",
        "Who is the headmaster of Hogwarts?",
        "Tell me about Voldemort.",
        "Who are Harry's best friends?",
        "What is a Horcrux?",
        "Who wrote the Harry Potter books?",
        "Is Snape good or bad?",
        "What is Quidditch?",
        "Where is Platform 9 3/4?",
        "Harry Potter and the Sorcerer's Stone",
        "The story of Harry Potter",
        "Hermione Granger and Ron Weasley",
        "Albus Dumbledore",
        "Draco Malfoy",
        "The Prisoner of Azkaban plot",
        "Severus Snape's secret",
        "Dobby the house elf",
        "The Battle of Hogwarts",
        "Fantastic Beasts and Where to Find Them"
    ]

    # --- 2. 负样本 (Safe): 通用常识问题 ---
    # 为了提升实验的权威性，本实践使用 SQuAD (Stanford Question Answering Dataset) 
    # 作为"通用/安全"知识的来源。
    negative_samples = []
    try:
        from datasets import load_dataset
        print("Loading SQuAD dataset from HuggingFace for negative samples...")
        # 加载前 200 条数据作为负样本
        dataset = load_dataset("squad", split="train[:200]")
        # 过滤掉可能包含 Harry Potter 的巧合 (虽然概率极低)
        for item in dataset:
            q = item["question"]
            if "Harry" not in q and "Potter" not in q:
                negative_samples.append(q)
        print(f"Successfully loaded {len(negative_samples)} samples from SQuAD.")
    except Exception as e:
        print(f"Warning: Failed to load SQuAD dataset ({e}). Using fallback data.")
        # 回退方案：手动构造的通用问题
        negative_samples = [
            "What is the capital of France?",
            "How do I boil an egg?",
            "Who is the president of the USA?",
            "What is the speed of light?",
            "Tell me a joke.",
            "How to write a python script?",
            "What is the weather like today?",
            "Who won the World Cup?",
            "Explain quantum physics.",
            "What is a neural network?",
            "How to bake a cake",
            "The history of China",
            "Basic math problems",
            "Learn to play guitar",
            "Travel tips for Japan",
            "What is the population of Earth?",
            "How does a car engine work?",
            "Who wrote Romeo and Juliet?",
            "What is the largest ocean?",
            "Definition of artificial intelligence"
        ]

    # 确保正负样本平衡 (虽然 SVM 对不平衡有一定容忍度，但平衡更好)
    # 如果 SQuAD 加载了太多，本实践截取一部分，或者通过 class_weight='balanced' 处理
    # 这里简单截取，保持大约 1:5 的比例即可，让负样本多一些代表通用性
    if len(negative_samples) > 100:
        negative_samples = negative_samples[:100]

    X = positive_samples + negative_samples
    y = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    print(f"Dataset size: {len(X)} (Positive: {len(positive_samples)}, Negative: {len(negative_samples)})")

    # 构建管道：TF-IDF 特征提取 -> SVM 分类器
    # 使用 class_weight='balanced' 自动处理样本不平衡
    clf = make_pipeline(
        TfidfVectorizer(stop_words='english'), # 去除停用词，关注实词
        SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    )
    clf.fit(X, y)

    # --- [Pattern Recognition Course Requirement] ---
    # 保存分类器的训练统计信息，用于可视化分析
    try:
        vectorizer = clf.named_steps['tfidfvectorizer']
        svm = clf.named_steps['svc']
        feature_names = vectorizer.get_feature_names_out()
        coefs = svm.coef_.toarray()[0]
        
        # 获取权重最高的特征 (最能代表"敏感"类别的词)
        top_k = 10
        top_indices = coefs.argsort()[-top_k:][::-1]
        top_features = [{"feature": feature_names[i], "weight": float(coefs[i])} for i in top_indices]
        
        stats = {
            "dataset_size": len(X),
            "positive_samples": len(positive_samples),
            "negative_samples": len(negative_samples),
            "top_sensitive_features": top_features
        }
        
        with open("classifier_stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        print("Classifier statistics saved to classifier_stats.json")
    except Exception as e:
        print(f"Could not save classifier stats: {e}")

    return clf

# --- 1.6 运行门控检查 (Gatekeeping Check) ---
# 实例化并训练分类器
gatekeeper = train_gatekeeper()

# 对当前用户查询进行推理，获取属于"敏感类"(索引1)的概率
is_sensitive_prob = gatekeeper.predict_proba([USER_QUERY])[0][1]
print(f"Query: '{USER_QUERY}'")
print(f"Sensitivity Score: {is_sensitive_prob:.4f}")

# 设定阈值为 0.5。如果概率低于阈值，视为安全查询，无需遗忘干预。
if is_sensitive_prob < 0.5:
    print("Query is SAFE. Skipping ECO optimization.")
    sys.exit(0)
else:
    print("Query is SENSITIVE. Initiating ECO Unlearning process...")

# --- 1.7 加载大语言模型 (LLM Loading) ---
print(f"Loading model: {MODEL_NAME}...")
# 使用 eco 库封装的 HFModel 类加载模型 (Qwen1.5-4B-Chat)
# config_path 指向模型配置文件，generation_config 设定生成参数
model = HFModel(
    model_name=MODEL_NAME, 
    config_path="config/model_config",
    generation_config=GenerationConfig(
        do_sample=False, max_new_tokens=256, use_cache=True
    )
)

# 获取分词器并处理 pad_token 缺失的常见问题
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
# 本实践手动查找包含敏感关键词的 Token 索引
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

# Prompt 包含系统提示和用户查询。
# 但模型生成的是 ANSWER。
# 腐蚀应该应用于 PROMPT Token，以便在模型开始生成答案之前破坏其内部状态。
# 然而，Qwen 的聊天模板可能会将 "Harry Potter" 部分放在后面。
# 此外，对于 Qwen/GPT 模型，有时 "Harry Potter" 中的 "Harry" 会被拆分或带有前缀。
# 因此需要确保捕获到了它们并验证

print(f"生成的 Mask: {mask}")
if sum(mask) == 0:
    print("警告: 没有 Token 被 Mask！优化将失败。")

# 用于概率计算的目标 ID
target_ids = tokenizer(TARGET_ANSWER, return_tensors="pt").input_ids.to(model.device)
target_id = target_ids[0, 0] # "Harry" 的第一个 Token

# --- 3. 优化循环 ---

def objective_function(strength, model, input_ids, target_id, mask):
    # 1. 应用腐蚀 Hook
    # 本实践直接使用底层 API，类似于 demo.py
    # 关键修复：'pos' 期望一个列表的列表 (batch_size, seq_len)
    # 并且它必须与 input_ids 的长度完全匹配。
    
    # 调试：打印攻击模块以确保本实践 Hook 到了正确的东西
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
    # 本实践必须确保 Hook 实际上被触发了。
    # Hook 注册在 Embedding 层上。
    # 当本实践调用 model.model(input_ids) 时，Embedding 层会被调用。
    
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids)
        # 输出 logits 对应于每个位置的下一个 Token 的预测。
        # 本实践想要 Prompt 结束后的那个 Token 的预测。
        # 所以本实践看序列中最后一个 Token 的 logits。
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
# 调整：为了获得平滑的下降曲线，本实践大幅降低学习率，并使用较小的维度
lr = 15  # 降低 LR，让它慢慢走
initial_strength = 0.1 # 从很小的噪声开始
eps = 0.05 
beta = initial_strength * 0.999
min_beta = 0.1
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
