import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Load dataset
dataset = load_dataset("json", data_files="finetune_data.jsonl", split="train")

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",  # Force CPU
    torch_dtype=torch.float32  # Use full precision on CPU
)

# Tokenize datasetIGNORE_INDEX = -100
IGNORE_INDEX = -100
def tokenize_function(example):
    prompt = example["prompt"]
    completion = example["completion"]
    prompt_plus_response = f"### Instruction:\n{prompt}\n\n### Response:\n{completion}"

    tokenized = tokenizer(prompt_plus_response, truncation=True, padding="max_length", max_length=512)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    labels = input_ids.copy()
    
    # Mask the prompt part
    prompt_only = f"### Instruction:\n{prompt}\n\n### Response:\n"
    prompt_tokenized = tokenizer(prompt_only, truncation=True, padding="max_length", max_length=512)
    prompt_len = len(prompt_tokenized["input_ids"])

    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Setup LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama_memories_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    warmup_steps=10,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    bf16=False,
    fp16=True,   # if you have fp16 GPU (else set to False)
    push_to_hub=False,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Start fine-tuning
trainer.train()

# Save the final model
model.save_pretrained("./tinyllama_memories_finetuned")
tokenizer.save_pretrained("./tinyllama_memories_finetuned")

print("✅ Fine-tuning complete! Model saved at './tinyllama_memories_finetuned'")
