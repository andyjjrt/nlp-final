from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import Dataset
import pandas as pd
import argparse, os
import torch
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser(description="Train model using LORA")
parser.add_argument(
    "--model",
    type=str,
    help="Target model",
    choices=["270M", "450M", "1_1B", "3B"],
    default="270M",
)
parser.add_argument(
    "--tokenizer", type=str, help="Tokenizer", default="meta-llama/Llama-2-7b-hf"
)
parser.add_argument(
    "--dataset",
    help="Dataset size",
    choices=["1k", "5k", "6k", "10k", "16k", "gpt"],
    default="1k",
)
parser.add_argument("--r", type=int, help="Lora Config r", default=32)
parser.add_argument("--lora_alpha", type=int, help="Lora Config lora_alpha", default=32)
parser.add_argument(
    "--lora_dropout", type=int, help="Lora Config lora_dropout", default=0.1
)
parser.add_argument(
    "--batch_size", type=int, help="Batch size, if your vram is low, use 1", default=4
)
parser.add_argument("--q4", help="Use bnbq4", action="store_true")

args = parser.parse_args()

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    f"apple/OpenELM-{args.model}-Instruct",
    token=HF_TOKEN,
    trust_remote_code=True,
    quantization_config=(
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        if args.q4
        else None
    ),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer, token=HF_TOKEN, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token


# Preproccess dataset data, transform to LORA dataset type
def process_func(example):
    new_example = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    instruction = tokenizer(
        f"<s>[INST]<SYS>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information.</SYS>{example['abstract']}[/INST]",
        add_special_tokens=False,
        truncation=True,
        max_length=512,
        padding=False,
    )
    response = tokenizer(
        f"{example['methods']}</s>",
        add_special_tokens=False,
        truncation=True,
        max_length=512,
        padding=False,
    )
    new_example["input_ids"] = instruction["input_ids"] + response["input_ids"]
    new_example["attention_mask"] = (
        instruction["attention_mask"] + response["attention_mask"]
    )
    new_example["labels"] = [-100] * len(instruction["input_ids"]) + response[
        "input_ids"
    ]
    return new_example


train_dataset = pd.read_csv(f"data/{args.dataset}/train.csv")
train_dataset = Dataset.from_pandas(train_dataset)
train_dataset = train_dataset.map(
    process_func, remove_columns=train_dataset.column_names
)
# train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= 512)
dataset = train_dataset.train_test_split(0.02)

print(train_dataset)

# Define LORA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear",
    inference_mode=False,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

output_name = f"loraV2_sft_{args.model}_{args.dataset}"

train_args = TrainingArguments(
    output_dir=f"./steps/{output_name}",
    learning_rate=1e-4,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    logging_steps=10,
    eval_strategy="steps",
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model),
)

trainer.train()

trainer.model.save_pretrained(f"./output/{output_name}")
tokenizer.save_pretrained(f"./output/{output_name}")