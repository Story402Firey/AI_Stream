"""
Trainer skeleton — supervised fine-tuning using Hugging Face Transformers + PEFT (LoRA).

This is a minimal example. Real training requires:
- A proper dataset with prompt/response formatting
- Tokenizer + model selection (GPU, accelerator setup)
- Validation split, checkpointing, hyperparameter tuning
- Mixed-precision + bitsandbytes for large models

The script below is intentionally minimal: adapt to your chosen model and infra.
"""
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import os

def make_dataset(jsonl_path):
    # Simple load via datasets
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    return ds

def preprocess_examples(example, tokenizer, max_length=1024):
    # Combine prompt and response into a single supervised text
    # Format depends on model; here's a simple "prompt + response" join
    text = f"User: {example['prompt']}\nAssistant: {example['response']}"
    tokenized = tokenizer(text, truncation=True, max_length=max_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main(args):
    ds = make_dataset(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", load_in_8bit=False)

    tokenized = ds.map(lambda ex: preprocess_examples(ex, tokenizer), remove_columns=ds.column_names)
    tokenized.set_format(type="numpy")
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiences.jsonl", help="JSONL training data")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face model to fine-tune")
    parser.add_argument("--output-dir", default="models/latest", help="Output directory")
    args = parser.parse_args()
    main(args)