# Diary to JSONL Project

This project consists of several Python scripts that process diary entries, fine-tune a language model, and allow you to chat with the fine-tuned model.

## Project Description

The main goal of this project is to convert diary entries into a structured JSONL format, fine-tune a TinyLlama language model with this data, and then use the fine-tuned model to generate responses based on the diary content.

## Files Description

*   `diary.txt`: Input file containing diary entries.
*   `memories.jsonl`: Output file in JSONL format, generated from `diary.txt`.
*   `finetune_data.jsonl`: Dataset used for fine-tuning the TinyLlama model, generated from `memories.jsonl`.
*   `finetune_tinyllama.py`: Script to fine-tune the TinyLlama model.
*   `generate_jsonl_smart.py`: Script to process `diary.txt` and generate `memories.jsonl`.
*   `embed_memories.py`: Script to embed memories and store them in a database.
*   `merge_and_chat.py`: Script to merge the fine-tuned model and chat with it.
*   `prepare_finetune_dataset.py`: Script to prepare the fine-tuning dataset from `memories.jsonl`.
*   `requirements.txt`: List of Python dependencies.
*   `finetuned/`: Directory containing fine-tuned model files.
*   `finetuned/chat_with_model.py`: Script to chat with the fine-tuned model.
*   `finetuned/merge_model.py`: Script to merge the LoRA adapters with the base model.
*   `memory_embeddings.db`: Database file storing memory embeddings.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download NLTK Resources:**

    If you encounter a `LookupError` when running `generate_jsonl_smart.py`, download the required NLTK resource:

    ```bash
    python -c "import nltk; nltk.download('punkt_tab')"
    ```

## Usage

### 1. Generate `memories.jsonl`

This script processes `diary.txt` and generates `memories.jsonl`.

```bash
python generate_jsonl_smart.py
```

### 2. Prepare Fine-tuning Dataset

This script prepares the fine-tuning dataset from `memories.jsonl` and saves it to `finetune_data.jsonl`.

```bash
python prepare_finetune_dataset.py
```

### 3. Fine-tune TinyLlama

This script fine-tunes the TinyLlama model using `finetune_data.jsonl`.

```bash
python finetune_tinyllama.py
```

### 4. Merge and Chat with the Model

This script merges the fine-tuned model and allows you to chat with it.

```bash
python merge_and_chat.py
```

Alternatively, you can use the scripts in the `finetuned/` directory:

*   `finetuned/merge_model.py`: Merges the LoRA adapters with the base model.
*   `finetuned/chat_with_model.py`: Chat with the fine-tuned model.

## Additional Notes

*   Ensure that `diary.txt` is properly formatted with diary entries separated by double newlines.
*   The fine-tuned model is saved in the `./tinyllama_memories_finetuned` directory.
