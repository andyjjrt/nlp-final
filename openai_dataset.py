import pandas as pd
import argparse, logging
from dotenv import load_dotenv
from openai import OpenAI
from utils import SimpleCSVHandler

load_dotenv()

parser = argparse.ArgumentParser(description="Make a verified dataset by GPT4o")
parser.add_argument(
    "--start", type=int, help="Start index", default=0
)
parser.add_argument(
    "--end", type=int, help="End index", default=99
)
args = parser.parse_args()

client = OpenAI()

dataset = pd.read_csv(f"data/eval.csv")

end = args.end
if end > len(dataset):
    end = len(dataset)

openai_csv = SimpleCSVHandler(f"data/eval_gpt4.csv")
openai_csv.write_header(["id", "abstract", "methods"])

for i in range(args.start, args.end):
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
    openai_csv.write_row([dataset.loc[i, "id"], dataset.loc[i, "abstract"], methods])
    print(methods)
    print("")

openai_csv.close()