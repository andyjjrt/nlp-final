import pandas as pd
import argparse, os, sys

parser = argparse.ArgumentParser(description="Merge dataset")
parser.add_argument(
    "--size", required=True, help="Dataset size", choices=["1k", "5k", "10k", "6k", "16k", "gpt"]
)
args = parser.parse_args()

if args.size == "gpt":
    gpt_data = pd.DataFrame()
    for f in os.listdir("data/output"):
        if f.startswith(f"dataset_gpt_"):
            df = pd.read_csv(f"data/output/{f}")
            gpt_data = pd.concat([gpt_data, df], ignore_index=True)
    gpt_data = gpt_data.drop(columns=["methods_llama3"])
    gpt_data.to_csv(f"data/train_gpt.csv", index=False)
    sys.exit()

concat = None
if args.size == "6k":
    concat = ["1k", "5k"]
elif args.size == "16k":
    concat = ["1k", "5k", "10k"]

if not os.path.exists(f"data/{args.size}"):
    os.mkdir(f"data/{args.size}")

if concat:
    merged_data = pd.DataFrame()
    for c in concat:
        df = pd.read_csv(f"data/{c}/train.csv")
        merged_data = pd.concat([merged_data, df], ignore_index=True)
    merged_data.to_csv(f"data/{args.size}/train.csv", index=False)
    sys.exit()

# Create an empty DataFrame to store merged data
success_data = pd.DataFrame()
fail_data = pd.DataFrame()
gpt_data = pd.DataFrame()

# Merge CSV files
for f in os.listdir("data/output"):
    if f.startswith(f"dataset_{args.size}_success"):
        df = pd.read_csv(f"data/output/{f}")
        success_data = pd.concat([success_data, df], ignore_index=True)
    if f.startswith(f"dataset_{args.size}_fail"):
        df = pd.read_csv(f"data/output/{f}")
        fail_data = pd.concat([fail_data, df], ignore_index=True)
    if f.startswith(f"dataset_gpt_{args.size}"):
        df = pd.read_csv(f"data/output/{f}")
        gpt_data = pd.concat([gpt_data, df], ignore_index=True)

# Save the merged DataFrame to a new CSV file
success_data.to_csv(f"data/train_{args.size}.csv", index=False)
fail_data.to_csv(f"data/train_{args.size}_fail.csv", index=False)
gpt_data.to_csv(f"data/train_{args.size}_gpt.csv", index=False)
    
size = int(str(args.size).replace("k", "000"))
gpt_data = gpt_data.drop(columns=["methods_llama3"])
merged_data = pd.concat([success_data.head(int(size * 0.9)), gpt_data.head(int(size * 0.1))], ignore_index=True)
merged_data.to_csv(f"data/{args.size}/train.csv", index=False)