import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os

# Check if we're on Apple Silicon or have CUDA available
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
use_quantization = torch.cuda.is_available()  # Only use quantization on CUDA

print(f"Using device: {device}")
print(f"Quantization enabled: {use_quantization}")

# 1. Define Model and Dataset Names
model_id = "microsoft/Phi-3-mini-4k-instruct"
dataset_file = "train_dataset.jsonl"
new_model_name = "phi-3-mini-kantra-rules-generator"

# 2. Configure Quantization for Memory Efficiency (QLoRA) - Only on CUDA
bnb_config = None
if use_quantization:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# 3. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. Load the Base Model with hardware-specific arguments
model_kwargs = {
    "trust_remote_code": True,
    "attn_implementation": "eager",  # Standard attention for broad compatibility
}

if bnb_config is not None:
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["device_map"] = "auto"
else:
    # Set appropriate dtype for MPS or CPU
    model_kwargs["torch_dtype"] = torch.bfloat16 if device == "cpu" else torch.float32

model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 5. Configure PEFT (Parameter-Efficient Fine-Tuning) with LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# 6. Load the Dataset
dataset = load_dataset("json", data_files=dataset_file, split="train")

# 7. Set Up Training Arguments with hardware-specific settings
training_args = TrainingArguments(
    output_dir=f"./{new_model_name}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    optim="adamw_torch" if not use_quantization else "paged_adamw_32bit",
    fp16=False,  # Disable fp16
    bf16=True if device == "cuda" else False,  # Use bf16 on CUDA for better performance
    # Use MPS device if available
    use_mps_device=(device == "mps")
)

# 8. Create the SFTTrainer (Supervised Fine-Tuning Trainer)
# Use minimal parameters for maximum compatibility across TRL versions
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

# 9. Start the Training Process
print("Starting the fine-tuning process...")
trainer.train()
print("Fine-tuning complete!")

# 10. Save the Final Model
final_model_path = f"./{new_model_name}-final"
trainer.save_model(final_model_path)
print(f"Model saved to {final_model_path}")