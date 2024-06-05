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
parser.add_argument("--name", type=str, help="Output name", default="lora_all_default")
parser.add_argument(
    "--model", type=str, help="Target model", default="apple/OpenELM-1_1B-Instruct"
)
parser.add_argument(
    "--tokenizer", type=str, help="Tokenizer", default="meta-llama/Llama-2-7b-hf"
)
parser.add_argument("--r", type=int, help="Lora Config r", default=32)
parser.add_argument("--lora_alpha", type=int, help="Lora Config lora_alpha", default=32)
parser.add_argument(
    "--lora_dropout", type=int, help="Lora Config lora_dropout", default=0.1
)
parser.add_argument(
    "--batch_size", type=int, help="Batch size, if your vran is low, use 1", default=4
)
parser.add_argument(
    "--q4", help="Use bnbq4", action="store_true"
)

args = parser.parse_args()

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    token=HF_TOKEN,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    ) if args.q4 else None,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer, token=HF_TOKEN, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token


# Preproccess dataset data, transform to LORA dataset type
def process_func(example):
    MAX_LENGTH = 1024
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<s>[INST]<SYS>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information.</SYS>{example['abstract']}[/INST]",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['methods']}</s>", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_dataset = pd.read_csv("data/train.csv")
train_dataset = Dataset.from_pandas(train_dataset)
train_dataset = train_dataset.map(
    process_func, remove_columns=train_dataset.column_names
)

print(train_dataset)

# Define LORA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear",
    inference_mode=False,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    init_lora_weights=True,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

output_name = args.name

train_args = TrainingArguments(
    output_dir=f"./steps/{output_name}",
    learning_rate=1e-4,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    logging_steps=100,
    save_steps=200,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model),
)

trainer.train()

trainer.model.save_pretrained(f"./output/{output_name}")
tokenizer.save_pretrained(f"./output/{output_name}")
