from datasets import load_dataset
import pandas as pd
import os, csv

if not os.path.exists("data"):
  os.mkdir("data")

try:
  dataset = pd.read_csv("data/data.csv", low_memory=False)
except FileNotFoundError:
  dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
  dataset = dataset.to_pandas()
  dataset = dataset[["id", "abstract"]]
  dataset = dataset.loc[(dataset['abstract'].str.len() > 512) & (dataset['abstract'].str.len() < 2048)]
  dataset["abstract"] = dataset["abstract"].map(lambda x: str(x).replace("\n", "").replace("\r", ""))
  dataset = dataset.reset_index(drop=True)
  dataset.to_csv("data/data.csv", index=False, quoting=csv.QUOTE_ALL)

dataset_10k = dataset.sample(n=60000, random_state=42)
dataset = dataset.drop(dataset_10k.index)

dataset_5k = dataset.sample(n=30000, random_state=42)
dataset = dataset.drop(dataset_5k.index)

dataset_1k = dataset.sample(n=6000, random_state=42)
dataset = dataset.drop(dataset_1k.index)

dataset_val = dataset.sample(n=100, random_state=42)
dataset = dataset.drop(dataset_val.index)


dataset_10k.to_csv("data/data_10k.csv", index=False, quoting=csv.QUOTE_ALL)
dataset_5k.to_csv("data/data_5k.csv", index=False, quoting=csv.QUOTE_ALL)
dataset_1k.to_csv("data/data_1k.csv", index=False, quoting=csv.QUOTE_ALL)
dataset_val.to_csv("data/data_val.csv", index=False, quoting=csv.QUOTE_ALL)