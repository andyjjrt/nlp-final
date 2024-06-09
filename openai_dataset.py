import pandas as pd
import argparse, logging
from dotenv import load_dotenv
from openai import OpenAI
from utils import SimpleCSVHandler

load_dotenv()

parser = argparse.ArgumentParser(description="Process abstract to methods using GPT4o")
parser.add_argument(
    "--size", required=True, help="Dataset size, this script will only use 10%% of the data, required train_{size}_fail.csv", choices=["1k", "5k", "10k", "val"]
)
parser.add_argument(
    "--start", type=int, help="Start index", default=0
)
parser.add_argument(
    "--n", type=int, help="Split range of the file", default=10
)
args = parser.parse_args()

client = OpenAI()

dataset = pd.read_csv(f"data/train_{args.size}_fail.csv")

if args.size == "1k":
    length = 100
elif args.size == "5k":
    length = 500
elif args.size == "10k":
    length = 1000

dataset = dataset.head(length)

split = args.n
rounds = int(args.start / split)
openai_csv = None

for i in range(args.start, length):
    if i % split == 0:
        if openai_csv != None:
            openai_csv.close()
        openai_csv = SimpleCSVHandler(f"data/output/dataset_gpt_{args.size}_{rounds * split}_{(rounds + 1) * split}.csv")
        openai_csv.write_header(["id", "abstract", "methods", "methods_llama3"])
        rounds += 1
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information."},
            {"role": "user", "content": str(dataset.loc[i, "abstract"])},],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    methods = response.choices[0].message.content.replace("\n", "").replace("\r", "")
    openai_csv.write_row([dataset.loc[i, "id"], dataset.loc[i, "abstract"], methods, dataset.loc[i, "methods"]])
    print(methods)
    print("")

openai_csv.close()