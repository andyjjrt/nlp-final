import pandas as pd
import argparse, os

parser = argparse.ArgumentParser(description="Merge data and dump train/val dataset")
parser.add_argument(
    "--frac", type=float, help="Split frac of the file", default=0.95
)
parser.add_argument(
    "--num", type=int, help="Start index", default=5000
)
args = parser.parse_args()

print(args)

# Create an empty DataFrame to store merged data
merged_data = pd.DataFrame()

# Merge CSV files
if os.path.exists("data/output"):
    for f in os.listdir("data/output"):
        if f.startswith("dataset_filtered_"):
            df = pd.read_csv(f"data/output/{f}")
            merged_data = pd.concat([merged_data, df], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_data.to_csv('data/data.csv', index=False)
else:
    merged_data = pd.read_csv(f"data/data.csv")

dataset = merged_data.sample(n=merged_data.shape[0], random_state=42)
dataset = dataset.loc[(dataset['abstract'].str.len() > 512) & (dataset['abstract'].str.len() < 2048)]

dataset = dataset.head(args.num)
train_dataset = dataset.sample(frac=args.frac, random_state=42)
eval_dataset = dataset.drop(train_dataset.index) 

# Save the merged DataFrame to a new CSV file
train_dataset.to_csv('data/train.csv', index=False)
eval_dataset.to_csv('data/eval.csv', index=False)