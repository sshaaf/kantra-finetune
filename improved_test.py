#!/usr/bin/env python3
"""
Improved test script that matches the training data format exactly.
This should generate proper YAML by using prompts similar to training examples.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import yaml
import os

# Configuration
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "./phi-3-mini-kantra-rules-generator-final"

# Device detection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
dtype = torch.bfloat16 if device == "cuda" else torch.float32
device_map = "auto" if device == "cuda" else None

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=dtype,
    device_map=device_map,
    trust_remote_code=True,
)

# Load fine-tuned model
print("Loading fine-tuned model...")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("Model loaded successfully!")

def test_kantra_rule_generation(prompt, max_tokens=400, temperature=0.1):
    """
    Test rule generation with training-style prompts
    """
    print(f"\n🔍 Testing: {prompt}")
    print("-" * 60)
    
    # Format exactly like training data
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(inputs)
    
    with torch.no_grad():
        generated = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    output = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    
    # Extract response
    if "<|assistant|>" in output:
        response = output.split("<|assistant|>")[1].strip()
    else:
        input_text = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
        response = output.replace(input_text, "").strip()
    
    print("📝 Generated Response:")
    print(response)
    print()
    
    # Validate YAML
    try:
        parsed = yaml.safe_load(response)
        print("✅ Valid YAML!")
        if isinstance(parsed, dict) and 'ruleID' in parsed:
            print(f"✅ Contains ruleID: {parsed['ruleID']}")
        return True, response
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False, response

def main():
    print("🧪 KANTRA RULE GENERATION TEST")
    print("=" * 60)
    
    # Test prompts that match training data style exactly
    training_style_prompts = [
        # Exact style from training data
        "Create a rule to detect usage of sun.misc.Unsafe class.",
        
        # Similar patterns from training data
        "Write a rule to identify deprecated javax.xml.bind imports.",
        
        "Create a rule to find content matching `java.util.Date` in *.java files.",
        
        "I need a rule to detect XML content using XPath `//web-app`. Web application configuration.",
        
        "Create a rule to detect Spring Boot 1.x configuration patterns.",
    ]
    
    results = []
    
    for prompt in training_style_prompts:
        success, response = test_kantra_rule_generation(prompt)
        results.append((prompt, success, response))
        print("\n" + "="*60 + "\n")
    
    # Summary
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"✅ Successful YAML generation: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful < total:
        print("\n💡 RECOMMENDATIONS:")
        print("1. Try lower temperature (0.01-0.05)")
        print("2. Use more training epochs")
        print("3. Add system prompt during training")
        print("4. Use deterministic generation (temperature=0)")
    
    # Test with different temperatures
    if successful < total:
        print("\n🔬 TESTING DIFFERENT TEMPERATURES")
        print("=" * 60)
        
        test_prompt = "Create a rule to detect usage of sun.misc.Unsafe class."
        
        for temp in [0.01, 0.05, 0.0]:  # Very low temperatures
            print(f"\n🌡️  Temperature: {temp}")
            success, _ = test_kantra_rule_generation(test_prompt, temperature=temp)
            if success:
                print(f"✅ Success with temperature {temp}!")
                break

if __name__ == "__main__":
    main()
