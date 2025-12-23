import sys
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, GenerationConfig

# Add parent directory to path to import eco
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eco.model import HFModel
from eco.attack import AttackedModel
from eco.optimizer import ZerothOrderOptimizerScalar
from eco.utils import load_yaml

# --- 1. Mock Components ---

class DummyPromptClassifier:
    def __init__(self, sensitive_keyword="Harry Potter"):
        self.sensitive_keyword = sensitive_keyword

    def predict(self, prompts, threshold=0.5):
        # Return 1 if sensitive keyword is in prompt, else 0
        return [1 if self.sensitive_keyword.lower() in p.lower() else 0 for p in prompts]

# --- 2. Setup ---

MODEL_NAME = "Qwen1.5-4B-Chat"
SENSITIVE_KEYWORD = "Harry Potter"
TARGET_ANSWER = "Harry Potter"
# We will format this prompt using the chat template later
USER_QUERY = "Who is the main character of the book series written by J.K. Rowling?"

print(f"Loading model: {MODEL_NAME}...")
# Ensure config exists (we created it)
model = HFModel(
    model_name=MODEL_NAME, 
    config_path="../config/model_config",
    generation_config=GenerationConfig(
        do_sample=False, max_new_tokens=256, use_cache=True
    )
)

# Setup Dummy Classifier
print("Setting up Prompt Classifier...")
prompt_classifier = DummyPromptClassifier(sensitive_keyword="Rowling") # Trigger on "Rowling"

# Setup Attacked Model
# We use 'rand_noise_first_n' as in the example, or similar.
# Let's check available corrupt methods in eco/attack/corrupt.py if needed, 
# but 'rand_noise_first_n' was in the example.
CORRUPT_METHOD = "rand_noise_first_n"
CORRUPT_DIMS = 10 # Corrupt 10 dimensions
INITIAL_STRENGTH = 0.0 # Start with 0

print("Wrapping model with AttackedModel...")
attacked_model = AttackedModel(
    model=model,
    prompt_classifier=prompt_classifier,
    token_classifier=None, # Will default to corrupting all tokens
    corrupt_method=CORRUPT_METHOD,
    corrupt_args={"dims": CORRUPT_DIMS, "strength": INITIAL_STRENGTH},
    classifier_threshold=0.5,
)

tokenizer = model.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply Chat Template
PROMPT = tokenizer.apply_chat_template(
    [{"role": "user", "content": USER_QUERY}],
    tokenize=False,
    add_generation_prompt=True,
)

# --- 3. Optimization Loop ---

def get_prob_of_target(model, prompt, target):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_ids = tokenizer(target, return_tensors="pt").input_ids.to(model.device)
    
    # We only care about the first token of the target for simplicity in this demo
    # or we can compute perplexity. Let's do probability of the first token of target.
    # "Harry"
    target_id = target_ids[0, 0] 
    
    with torch.no_grad():
        # AttackedModel.generate expects 'prompts' (list of str) as first arg for classifier,
        # and then passes *args/**kwargs to underlying model.generate.
        # Underlying model.generate needs 'input_ids'.
        # So we pass prompts=[prompt] and input_ids=inputs.input_ids
        
        # However, here we want logits, so we use model() (forward pass).
        # AttackedModel.__call__ signature: (self, prompts, answers, *args, **kwargs)
        # It passes *args, **kwargs to self.model()
        
        outputs = model(
            prompts=[prompt], 
            answers=[target], # Used for token-level corruption logic if needed
            input_ids=inputs.input_ids
        )
        
        logits = outputs.logits[0, -1, :] # Logits for the next token
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[target_id].item()
        
    return target_prob

def generate_text(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # AttackedModel.generate(prompts, *args, **kwargs)
        # We must pass input_ids as a kwarg so it reaches the underlying model
        outputs = model.generate(
            prompts=[prompt],
            input_ids=inputs.input_ids, 
            max_new_tokens=20, 
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

logs = []

def objective_function(strength, model, prompt, target):
    # Update strength
    model.update_corrupt_args({"dims": CORRUPT_DIMS, "strength": strength})
    
    # Calculate probability of target (We want to minimize this)
    prob = get_prob_of_target(model, prompt, target)
    
    return prob # The optimizer minimizes this value

print("Starting Optimization...")

# Optimizer settings
lr = 5.0 # Learning rate
eps = 0.1 
beta = 0.0 # Initial value
min_beta = 0.0
num_steps = 20

optimizer = ZerothOrderOptimizerScalar(
    lr=lr, eps=eps, beta=beta, min_beta=min_beta
)

context = {
    "model": attacked_model,
    "prompt": PROMPT,
    "target": TARGET_ANSWER
}

# Initial State
initial_prob = get_prob_of_target(attacked_model, PROMPT, TARGET_ANSWER)
initial_text = generate_text(attacked_model, PROMPT)
print(f"Initial: Prob={initial_prob:.4f}, Text='{initial_text}'")
logs.append({
    "step": -1,
    "strength": 0.0,
    "prob": initial_prob,
    "generated": initial_text
})

for i in range(num_steps):
    # The optimizer step expects a function that returns a score (to be minimized? or maximized?)
    # In zeroth_order_optim.py:
    # output = optimizer.step(score, ...)
    # And it prints f_score.
    # Let's check optimizer code to see if it minimizes or maximizes.
    # Usually ZO minimizes loss.
    # If we want to UNLEARN, we want to MINIMIZE the probability of the target.
    
    output = optimizer.step(objective_function, context)
    
    current_strength = optimizer.beta
    current_prob = output["f_score"]
    
    # Generate text for logging
    current_text = generate_text(attacked_model, PROMPT)
    
    print(f"Step {i}: Strength={current_strength:.4f}, Prob={current_prob:.4f}")
    
    logs.append({
        "step": i,
        "strength": current_strength,
        "prob": current_prob,
        "generated": current_text
    })
    
    if current_prob < 0.01: # Early stop if probability is very low
        print("Target probability low enough. Stopping.")
        break

# Save logs
log_path = os.path.join(os.path.dirname(__file__), "demo_logs.json")
with open(log_path, "w") as f:
    json.dump(logs, f, indent=2)

print(f"Logs saved to {log_path}")
