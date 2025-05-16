from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import os

# Paths
adapter_path = "./adapter"
merged_model_path = "./merged_model"
base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Will download automatically if not present

# 1. Load TinyLlama base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float32,   # Use float32 on CPU
    device_map="cpu"
)

# 2. Load your LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# 3. Merge and unload
model = model.merge_and_unload()

# 4. Save merged full model
os.makedirs(merged_model_path, exist_ok=True)
model.save_pretrained(merged_model_path)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)

print("✅ Merged full fine-tuned model saved at:", merged_model_path)

# -------------------------------------
# Now start chatting
# -------------------------------------

# Reload merged model for chatting
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

# Chat loop
print("\n🧠 Memory AI ready! Start chatting. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    response = pipe(prompt)
    print("\nJarvis:", response[0]["generated_text"].replace(prompt, "").strip(), "\n")
