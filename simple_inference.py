#!/usr/bin/env python3
"""
Simple Inference Script for Merged Kantra Model

This script works with the merged model (no PEFT/LoRA complexity).
Much simpler and more reliable than adapter-based inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import re

# Configuration
model_path = "./kantra-phi3-merged"  # Path to merged model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print("🚀 SIMPLE KANTRA RULE INFERENCE")
print("=" * 50)
print(f"📱 Device: {device}")
print(f"📁 Model: {model_path}")

# System prompt (same as training)
SYSTEM_PROMPT = """You are a Kantra migration rule generator. You MUST respond with valid YAML only, starting with '---'. Do not include any conversational text, explanations, or markdown. Only output the YAML rule structure."""

def load_model():
    """Load the merged model and tokenizer"""
    print("\n📚 Loading model and tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        
        if device != "cuda":
            model = model.to(device)
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure you've run robust_finetune.py first to create the merged model")
        return None, None

def clean_yaml_response(response):
    """
    Clean and extract YAML from response.
    The robust training should minimize this need, but it's here as backup.
    """
    
    # Remove any leading conversational text
    if "---" in response:
        yaml_part = response.split("---", 1)[1].strip()
        return "---\n" + yaml_part
    
    # Look for YAML keys
    yaml_keys = ["ruleID:", "category:", "description:", "when:", "message:", "labels:"]
    for key in yaml_keys:
        if key in response:
            key_pos = response.find(key)
            return response[key_pos:].strip()
    
    return response.strip()

def generate_kantra_rule(model, tokenizer, prompt, max_tokens=400, temperature=0.1):
    """Generate a Kantra rule for the given prompt"""
    
    print(f"\n🔍 Prompt: {prompt}")
    print("-" * 50)
    
    # Format with system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    )
    inputs = inputs.to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Clean the response
    cleaned_response = clean_yaml_response(response)
    
    print("📝 Generated Rule:")
    print(cleaned_response)
    
    # Validate YAML
    try:
        parsed = yaml.safe_load(cleaned_response)
        if isinstance(parsed, dict):
            print(f"\n✅ Valid YAML generated!")
            
            # Show key information
            if 'ruleID' in parsed:
                print(f"   🆔 Rule ID: {parsed['ruleID']}")
            if 'description' in parsed:
                print(f"   📄 Description: {parsed['description']}")
            if 'category' in parsed:
                print(f"   🏷️  Category: {parsed['category']}")
            if 'effort' in parsed:
                print(f"   ⚡ Effort: {parsed['effort']}")
            
            return True, cleaned_response, parsed
        else:
            print("❌ Generated content is not a valid YAML object")
            return False, cleaned_response, None
            
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False, cleaned_response, None

def interactive_mode(model, tokenizer):
    """Interactive mode for testing multiple prompts"""
    
    print("\n🎯 INTERACTIVE MODE")
    print("=" * 50)
    print("Enter prompts to generate Kantra rules. Type 'quit' to exit.")
    print("Examples:")
    print("  - Create a rule to detect usage of sun.misc.Unsafe class.")
    print("  - Write a rule to identify deprecated javax.xml.bind imports.")
    print("  - Create a rule to find hardcoded database connections.")
    
    while True:
        try:
            prompt = input("\n💬 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            success, yaml_content, parsed = generate_kantra_rule(model, tokenizer, prompt)
            
            if success:
                # Ask if user wants to save the rule
                save = input("\n💾 Save this rule to file? (y/n): ").strip().lower()
                if save in ['y', 'yes']:
                    rule_id = parsed.get('ruleID', 'generated-rule')
                    filename = f"{rule_id}.yaml"
                    
                    with open(filename, 'w') as f:
                        f.write(yaml_content)
                    
                    print(f"✅ Saved to {filename}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_test_mode(model, tokenizer):
    """Test with a batch of predefined prompts"""
    
    test_prompts = [
        "Create a rule to detect usage of sun.misc.Unsafe class.",
        "Write a rule to identify deprecated javax.xml.bind imports.",
        "Create a rule to find hardcoded database connections in Java files.",
        "Write a rule to detect Spring Boot 1.x configuration patterns.",
        "Create a rule to identify usage of java.util.Date instead of java.time classes.",
        "Write a rule to find hardcoded file paths in configuration files.",
    ]
    
    print(f"\n🧪 BATCH TEST MODE")
    print("=" * 50)
    print(f"Testing {len(test_prompts)} prompts...")
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📋 Test {i}/{len(test_prompts)}")
        success, yaml_content, parsed = generate_kantra_rule(model, tokenizer, prompt)
        results.append((prompt, success, parsed))
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 50)
    print(f"✅ Successful: {successful}/{len(test_prompts)} ({successful/len(test_prompts)*100:.1f}%)")
    
    if successful > 0:
        print("\n🎉 SUCCESS! The robust training pipeline is working!")
        
        # Show successful examples
        print("\n📋 Generated Rule IDs:")
        for prompt, success, parsed in results:
            if success and parsed and 'ruleID' in parsed:
                print(f"   • {parsed['ruleID']}")

def main():
    # Load model
    model, tokenizer = load_model()
    
    if model is None:
        return
    
    # Choose mode
    print(f"\n🎯 Choose mode:")
    print("1. Interactive mode (enter prompts manually)")
    print("2. Batch test mode (test predefined prompts)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_mode(model, tokenizer)
        elif choice == "2":
            batch_test_mode(model, tokenizer)
        else:
            print("Invalid choice. Running batch test mode...")
            batch_test_mode(model, tokenizer)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()

