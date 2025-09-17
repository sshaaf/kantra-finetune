#!/usr/bin/env python3
"""
Robust Kantra Fine-tuning Script with Explicit Formatting and Model Merging

Key improvements:
1. Explicit chat template formatting during training
2. System prompt to enforce YAML-only output
3. Automatic LoRA adapter merging
4. Standalone model saving for easy inference
5. Better error handling and validation
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import json
import os
import yaml
from pathlib import Path

# Configuration
model_id = "microsoft/Phi-3-mini-4k-instruct"
dataset_path = "train_dataset.jsonl"
output_dir = "kantra-phi3-robust"
merged_model_dir = "kantra-phi3-merged"

# Training parameters
max_seq_length = 2048
batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4
num_epochs = 3
warmup_steps = 10

print("🚀 ROBUST KANTRA FINE-TUNING PIPELINE")
print("=" * 60)

# Device detection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
use_quantization = device == "cuda"

print(f"📱 Device: {device}")
print(f"🔧 Quantization: {use_quantization}")

# Load tokenizer
print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# System prompt to enforce YAML output
SYSTEM_PROMPT = """You are a Kantra migration rule generator. You MUST respond with valid YAML only, starting with '---'. Do not include any conversational text, explanations, or markdown. Only output the YAML rule structure."""

def format_training_example(example):
    """
    Explicitly format training examples with system prompt and chat template.
    This ensures the model learns the exact input/output format we want.
    """
    user_content = example["messages"][0]["content"]
    assistant_content = example["messages"][1]["content"]
    
    # Create properly formatted conversation
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}

def validate_yaml_content(content):
    """Validate that the content is proper YAML"""
    try:
        parsed = yaml.safe_load(content)
        return isinstance(parsed, dict) and 'ruleID' in parsed
    except:
        return False

# Load and validate dataset
print("\n📊 Loading and processing dataset...")
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Validate a few examples
    valid_count = 0
    for i in range(min(10, len(dataset))):
        assistant_content = dataset[i]["messages"][1]["content"]
        if validate_yaml_content(assistant_content):
            valid_count += 1
    
    print(f"✅ Validated {valid_count}/10 sample examples contain valid YAML")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Format dataset with explicit chat templates
print("🔧 Formatting dataset with chat templates...")
formatted_dataset = dataset.map(format_training_example, remove_columns=dataset.column_names)
print(f"✅ Formatted {len(formatted_dataset)} examples")

# Show a sample formatted example
print("\n📝 Sample formatted training example:")
print("-" * 40)
sample_text = formatted_dataset[0]["text"]
print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
print("-" * 40)

# Quantization config
bnb_config = None
if use_quantization:
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("✅ 4-bit quantization enabled")
    except ImportError:
        print("⚠️  bitsandbytes not available, using full precision")
        use_quantization = False

# Load model
print(f"\n🤖 Loading model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config if use_quantization else None,
    dtype=torch.bfloat16 if use_quantization else torch.float32,
    device_map="auto" if use_quantization else None,
    trust_remote_code=True,
    attn_implementation="eager",  # More compatible with different GPUs
)

# Prepare model for training
if use_quantization:
    model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
print("✅ LoRA adapter applied")

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to=[],  # Disable wandb
    optim="adamw_torch" if not use_quantization else "paged_adamw_8bit",
    fp16=False,
    bf16=use_quantization,
    dataloader_pin_memory=not use_quantization,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

# Create trainer
print("\n🎯 Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    tokenizer=tokenizer,
)

# Start training
print(f"\n🚀 Starting training for {num_epochs} epochs...")
print(f"📊 Total steps: {len(formatted_dataset) // (batch_size * gradient_accumulation_steps) * num_epochs}")

try:
    trainer.train()
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    exit(1)

# Save the LoRA adapter
print(f"\n💾 Saving LoRA adapter to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# MERGE THE ADAPTER INTO BASE MODEL
print(f"\n🔄 Merging LoRA adapter into base model...")

try:
    # Load the base model again (clean)
    print("📚 Loading clean base model for merging...")
    base_model_for_merge = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,  # Use fp16 for merging to save memory
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load and merge the adapter
    from peft import PeftModel
    print("🔧 Loading and merging adapter...")
    merged_model = PeftModel.from_pretrained(base_model_for_merge, output_dir)
    merged_model = merged_model.merge_and_unload()
    
    # Save the merged model
    print(f"💾 Saving merged model to {merged_model_dir}...")
    os.makedirs(merged_model_dir, exist_ok=True)
    merged_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    
    print("✅ Model merging completed successfully!")
    
    # Clean up GPU memory
    del base_model_for_merge, merged_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
except Exception as e:
    print(f"⚠️  Model merging failed: {e}")
    print("📝 LoRA adapter is still available for inference")

# Create a simple test
print(f"\n🧪 Testing merged model...")
try:
    # Load merged model for testing
    test_model = AutoModelForCausalLM.from_pretrained(
        merged_model_dir,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    test_tokenizer = AutoTokenizer.from_pretrained(merged_model_dir, trust_remote_code=True)
    test_tokenizer.pad_token = test_tokenizer.eos_token
    
    # Test prompt
    test_prompt = "Create a rule to detect usage of sun.misc.Unsafe class."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_prompt}
    ]
    
    inputs = test_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(test_model.device)
    
    with torch.no_grad():
        outputs = test_model.generate(
            inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=test_tokenizer.pad_token_id,
            eos_token_id=test_tokenizer.eos_token_id,
        )
    
    response = test_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    print("📝 Test response:")
    print(response[:200] + "..." if len(response) > 200 else response)
    
    # Validate YAML
    if validate_yaml_content(response):
        print("✅ Generated valid YAML!")
    else:
        print("⚠️  Generated content may need YAML extraction")
    
    # Clean up
    del test_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
except Exception as e:
    print(f"⚠️  Test failed: {e}")

print(f"\n🎉 TRAINING PIPELINE COMPLETE!")
print("=" * 60)
print(f"📁 LoRA Adapter: {output_dir}/")
print(f"📁 Merged Model: {merged_model_dir}/")
print(f"🚀 Use the merged model for simple inference!")

