# Kantra Fine-tune

Fine-tuning Phi-3 Mini to generate Kantra migration rules using LoRA.
The project is experiemental, and thats why you see alot of python scripts doing the same. 
This is a learning playground. 

## What this project does

This project fine-tunes Microsoft's Phi-3-mini model to automatically generate YAML-formatted Kantra migration rules. Kantra is a tool for modernizing applications, and this model helps create rules that detect deprecated APIs, patterns, and code that needs migration.

Goal: To train a small LLM with Kantra rules. 
- `train_dataset.jsonl` - Training dataset containing examples of user prompts and expected Kantra rule outputs in JSONL format. The dataset is taken from the 2400 rules in the Konveyor rulesets.

Each line in the dataset contains:
```json
{
  "text": "user prompt and assistant response in chat format"
}
```

## Files

### Notebooks
- `kantra_finetune.ipynb` - Original fine-tuning notebook using standard transformers
- `kantra_finetune_unsloth.ipynb` - Optimized fine-tuning using Unsloth (2x faster, 50% less memory)

### Python Scripts
- `kantra_finetune.py` - Python version of the original fine-tuning script
- `robust_finetune.py` - Enhanced fine-tuning with automatic model merging
- `simple_inference.py` - Simple script to test the fine-tuned model
- `kantra_testmodel.py` - Model testing utilities
- `fixed_test.py` - Fixed version of testing script
- `improved_test.py` - Improved testing with better error handling

## Usage

1. **Standard fine-tuning**: Use `kantra_finetune.ipynb`
2. **Optimized fine-tuning**: Use `kantra_finetune_unsloth.ipynb` (recommended)
3. **Testing**: Use `simple_inference.py` after training

