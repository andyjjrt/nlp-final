from datasets import load_dataset
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from utils import SimpleCSVHandler
import logging
import argparse, os

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S", level=logging.INFO
)

parser = argparse.ArgumentParser(description="Process abstract to methods using llama")
parser.add_argument(
    "--size", required=True, help="Dataset size", choices=["1k", "5k", "10k", "val"]
)
parser.add_argument(
    "--n", type=int, help="Split range of the file", default=100
)
parser.add_argument(
    "--index", type=int, help="Start index", default=0
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

dataset = load_dataset("csv", data_files=f"data/data_{args.size}.csv", split="train")
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, and NOTHING else.<|eot_id|><|start_header_id|>user<|end_header_id|>
```
{abstract}
```
Methods: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["abstract"],
)
model = Ollama(model="llama3:latest", stop=["<|eot_id|>"])
chain = prompt | model

split = args.n
satart_index = args.index
rounds = int(satart_index / split)

fail_csv = None
success_csv = None

length = len(dataset)

for i in range(satart_index, length):
    example = dataset[i]
    if i % split == 0:
        if fail_csv != None:
            fail_csv.close()
        if success_csv != None:
            success_csv.close()
        fail_csv = SimpleCSVHandler(f"data/output/dataset_{args.size}_fail_{rounds * split}_{(rounds + 1) * split}.csv")
        success_csv = SimpleCSVHandler(f"data/output/dataset_{args.size}_success_{rounds * split}_{(rounds + 1) * split}.csv")
        fail_csv.write_header(["id", "abstract", "methods"])
        success_csv.write_header(["id", "abstract", "methods"])
        rounds += 1

    abstract = example["abstract"].replace("\n", "").replace("\r", "")
    output = chain.invoke(abstract)
    
    methods = output.split("\n")
    methods = [m.removeprefix("* ") for m in methods if m.startswith("*")]
    answer = " ".join(methods)
    
    if len(methods) == 0:
        logging.warning(f"Error extract abstract {example['id']} ({i}): no method found")
        fail_csv.write_row([example['id'], abstract, output.replace("\n", "").replace("\r", "")])
    else:
        if is_method_in_abstract(abstract, answer):
            logging.info(f"Successfully extract abstract {example['id']} ({i})")
            success_csv.write_row([example['id'], abstract, answer])
        else:
            logging.warning(f"Error extract abstract {example['id']} ({i}): method not match")
            fail_csv.write_row([example['id'], abstract, answer])
