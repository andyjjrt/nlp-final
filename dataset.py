from datasets import load_dataset
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from utils import SimpleCSVHandler
import logging
import argparse, os

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S", level=logging.INFO
)

parser = argparse.ArgumentParser(description="Process abstract to methods")
parser.add_argument(
    "--split_count", type=int, help="Split range of the file"
)
parser.add_argument(
    "--start_index", type=int, help="Start index", default=0
)
args = parser.parse_args()

# Basic parser to extract markdown bullet list
def parse(output: str) -> str:
    methods = output.split("\n")
    methods = [m.removeprefix("* ") for m in methods if m.startswith("*")]
    if len(methods) == 0:
        return ""
    return " ".join(methods)

def is_method_in_abstract(abstract, methods):
    # 移除空格
    abstract = abstract.replace("\n", "").replace("\r", "").replace(" ", "")
    method = methods.replace("\n", "").replace("\r", "").replace(" ", "")
    return method in abstract

if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isdir("data/output"):
    os.mkdir("data/output")

dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, and NOTHING else.<|eot_id|><|start_header_id|>user<|end_header_id|>
```
{abstract}
```
Methods: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["abstract"],
)
model = Ollama(model="llama3:latest", stop=["<|eot_id|>"])
chain = prompt | model | parse

split = args.split_count
satart_index = args.start_index
rounds = int(satart_index / split)

generated_csv = None
filtered_csv = None

length = len(dataset)

for i in range(satart_index, length):
    example = dataset[i]
    if i % split == 0:
        if generated_csv != None:
            generated_csv.close()
        if filtered_csv != None:
            filtered_csv.close()
        generated_csv = SimpleCSVHandler(f"output/dataset_{rounds * split}_{(rounds + 1) * split}.csv")
        filtered_csv = SimpleCSVHandler(f"output/dataset_filtered_{rounds * split}_{(rounds + 1) * split}.csv")
        generated_csv.write_header(["id", "abstract", "methods"])
        filtered_csv.write_header(["id", "abstract", "methods"])
        rounds += 1

    abstract = example["abstract"].replace("\n", "").replace("\r", "")
    methods = chain.invoke(abstract)
    if len(methods) != 0:
        logging.info(f"Successfully extract abstract {example['id']} ({i})")
        generated_csv.write_row([example['id'], abstract, methods])
        if is_method_in_abstract(abstract, methods):
            filtered_csv.write_row([example['id'], abstract, methods])
    else:
        logging.warning(f"Error extract abstract {example['id']} ({i}): no method found")
