import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

# Detect device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the base model and your fine-tuned adapter
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "./phi-3-mini-kantra-rules-generator-final"

# Load the base model tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Set appropriate dtype for MPS or CPU
model_dtype = torch.bfloat16 if device == "cpu" else torch.float32

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # Fixed deprecated torch_dtype
    device_map="auto" if device == "cuda" else None,  # Use auto for CUDA, None for others
    trust_remote_code=True,
    attn_implementation="eager",
)

# --- THIS IS THE KEY STEP ---
# Load the PEFT model by merging the adapter weights into the base model
print("Loading adapter and merging with the base model...")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("Merge complete.")

# --- INFERENCE ---
# Define a new prompt that the model has NOT seen during training
# Use a more specific prompt that matches the training format
new_prompt = """Create a Kantra migration rule in YAML format to detect when a Java file imports `sun.misc.Unsafe`, which is a non-portable and risky API that should be avoided.

Please provide the rule in the following YAML structure:
- ruleID: (unique identifier)
- when: (condition to match)
- perform: (action to take)
- message: (description of the issue)"""

# Format the prompt using the chat template
messages = [
    {"role": "user", "content": new_prompt},
]
# Manually move inputs to the correct device
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

print("\nGenerating response for new prompt...")
# Generate the output with proper attention mask and generation config
attention_mask = torch.ones_like(model_inputs)
generated_ids = model.generate(
    model_inputs,
    attention_mask=attention_mask,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False,  # Disable cache to avoid DynamicCache issues
)

# Decode the generated tokens into text
decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Extract only the assistant's response
if "<|assistant|>" in decoded_output:
    response = decoded_output.split("<|assistant|>")[1].strip()
else:
    # Fallback: remove the input prompt from the output
    input_text = tokenizer.batch_decode(model_inputs, skip_special_tokens=True)[0]
    response = decoded_output.replace(input_text, "").strip()

print("\n--- MODEL OUTPUT ---")
print(response)
print("--------------------\n")

# --- VALIDATION ---
try:
    yaml.safe_load(response)
    print("✅ Success! The output is valid YAML.")
except yaml.YAMLError as e:
    print(f"❌ Error: The output is not valid YAML.\n{e}")

print("\n" + "="*60)
print("🧪 TESTING ADDITIONAL PROMPTS")
print("="*60)

# Test with different prompts to see the model's behavior
test_prompts = [
    "Generate a YAML Kantra rule for detecting deprecated javax.xml.bind imports",
    "Create a Kantra rule to identify usage of java.util.Date instead of java.time.LocalDate",
    "Write a YAML rule to detect Spring Boot 1.x configuration patterns"
]

for i, test_prompt in enumerate(test_prompts, 1):
    print(f"\n🔍 Test {i}: {test_prompt[:50]}...")
    
    test_messages = [{"role": "user", "content": test_prompt}]
    test_inputs = tokenizer.apply_chat_template(test_messages, return_tensors="pt").to(device)
    test_attention_mask = torch.ones_like(test_inputs)
    
    with torch.no_grad():
        test_generated = model.generate(
            test_inputs,
            attention_mask=test_attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    test_output = tokenizer.batch_decode(test_generated, skip_special_tokens=True)[0]
    
    if "<|assistant|>" in test_output:
        test_response = test_output.split("<|assistant|>")[1].strip()
    else:
        test_input_text = tokenizer.batch_decode(test_inputs, skip_special_tokens=True)[0]
        test_response = test_output.replace(test_input_text, "").strip()
    
    print(f"Response: {test_response[:200]}...")
    
    # Quick YAML validation
    try:
        yaml.safe_load(test_response)
        print("✅ Valid YAML")
    except:
        print("❌ Invalid YAML")

print("\n💡 If the model is not generating proper YAML, consider:")
print("   1. Checking the training data format")
print("   2. Using more specific prompts")
print("   3. Fine-tuning for more epochs")
print("   4. Adjusting the LoRA parameters")