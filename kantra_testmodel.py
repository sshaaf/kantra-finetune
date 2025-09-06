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
new_prompt = "Generate a Kantra rule to detect when a Java file imports `sun.misc.Unsafe`, which is a non-portable and risky API."

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