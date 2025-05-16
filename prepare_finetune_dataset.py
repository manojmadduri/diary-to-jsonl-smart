import json

input_path = 'memories.jsonl'
output_path = 'finetune_data.jsonl'

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        mem = json.loads(line)
        prompt = mem['input']
        completion = mem['output']
        outfile.write(json.dumps({
            "prompt": prompt,
            "completion": completion
        }) + "\n")

print(f"✅ Fine-tuning dataset created at '{output_path}'")
