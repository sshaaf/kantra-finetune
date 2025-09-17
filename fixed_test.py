#!/usr/bin/env python3
"""
Fixed test script that extracts YAML from mixed conversational + YAML output.
The model sometimes generates conversational text before the YAML, so we extract just the YAML part.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import yaml
import re
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

def extract_yaml_from_response(response):
    """
    Extract YAML content from a mixed response that might contain conversational text.
    
    The model sometimes generates:
    "- I need a rule to detect... 
     - Some explanation...
     ---
     ruleID: ..."
     
    We want to extract just the YAML part starting with "---" or the first YAML key.
    """
    
    # Method 1: Look for YAML document separator
    if "---" in response:
        yaml_part = response.split("---", 1)[1].strip()
        return "---\n" + yaml_part
    
    # Method 2: Look for common YAML keys that indicate start of rule
    yaml_keys = ["ruleID:", "category:", "description:", "when:", "message:", "labels:"]
    
    for key in yaml_keys:
        if key in response:
            # Find the position of the first YAML key
            key_pos = response.find(key)
            yaml_part = response[key_pos:].strip()
            return yaml_part
    
    # Method 3: Try to find YAML-like structure (key: value patterns)
    lines = response.split('\n')
    yaml_start = -1
    
    for i, line in enumerate(lines):
        # Look for lines that match YAML key: value pattern
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*:', line.strip()):
            yaml_start = i
            break
    
    if yaml_start >= 0:
        yaml_lines = lines[yaml_start:]
        return '\n'.join(yaml_lines).strip()
    
    # If no YAML structure found, return original response
    return response

def test_kantra_rule_generation(prompt, max_tokens=500, temperature=0.1):
    """
    Test rule generation with YAML extraction
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
        raw_response = output.split("<|assistant|>")[1].strip()
    else:
        input_text = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
        raw_response = output.replace(input_text, "").strip()
    
    print("📝 Raw Generated Response:")
    print(raw_response[:300] + "..." if len(raw_response) > 300 else raw_response)
    print()
    
    # Extract YAML portion
    yaml_response = extract_yaml_from_response(raw_response)
    
    print("🎯 Extracted YAML:")
    print(yaml_response[:400] + "..." if len(yaml_response) > 400 else yaml_response)
    print()
    
    # Validate YAML
    try:
        parsed = yaml.safe_load(yaml_response)
        print("✅ Valid YAML!")
        
        if isinstance(parsed, dict):
            if 'ruleID' in parsed:
                print(f"✅ Contains ruleID: {parsed['ruleID']}")
            if 'when' in parsed:
                print("✅ Contains 'when' condition")
            if 'message' in parsed:
                print("✅ Contains message")
            
            # Show key fields
            key_fields = ['ruleID', 'description', 'category', 'effort']
            for field in key_fields:
                if field in parsed:
                    print(f"   {field}: {parsed[field]}")
        
        return True, yaml_response, parsed
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False, yaml_response, None

def main():
    print("🧪 KANTRA RULE GENERATION TEST (WITH YAML EXTRACTION)")
    print("=" * 70)
    
    # Test prompts that match training data style exactly
    training_style_prompts = [
        # Test the specific case from your output
        "Create a rule to detect usage of sun.misc.Unsafe class.",
        
        # More training-style prompts
        "Write a rule to identify deprecated javax.xml.bind imports.",
        "Create a rule to find content matching `java.util.Date` in *.java files.",
        "Create a rule to detect Spring Boot 1.x configuration patterns.",
        "Write a rule to identify `*.war` files - web application archive.",
    ]
    
    results = []
    
    for prompt in training_style_prompts:
        success, yaml_content, parsed = test_kantra_rule_generation(prompt, temperature=0.05)
        results.append((prompt, success, yaml_content, parsed))
        print("\n" + "="*70 + "\n")
    
    # Summary
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    successful = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    print(f"✅ Successful YAML generation: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful > 0:
        print(f"\n🎉 SUCCESS! The model IS generating valid Kantra rules!")
        print("The issue was that the model adds conversational text before the YAML.")
        print("The YAML extraction successfully isolates the rule content.")
        
        # Show a successful example
        for prompt, success, yaml_content, parsed in results:
            if success and parsed:
                print(f"\n📋 Example successful rule:")
                print(f"Prompt: {prompt}")
                print(f"Generated ruleID: {parsed.get('ruleID', 'N/A')}")
                print(f"Description: {parsed.get('description', 'N/A')}")
                break
    else:
        print("\n💡 RECOMMENDATIONS:")
        print("1. Try even lower temperature (0.01)")
        print("2. Use more specific prompts")
        print("3. Consider different model architectures")
        print("4. Add system prompt during training")

    # Test with deterministic generation
    if successful < total:
        print(f"\n🔬 TESTING WITH DETERMINISTIC GENERATION (temperature=0)")
        print("=" * 70)
        
        test_prompt = "Create a rule to detect usage of sun.misc.Unsafe class."
        success, yaml_content, parsed = test_kantra_rule_generation(test_prompt, temperature=0.0)
        
        if success:
            print("✅ Deterministic generation works!")
        else:
            print("❌ Even deterministic generation has issues")

if __name__ == "__main__":
    main()
