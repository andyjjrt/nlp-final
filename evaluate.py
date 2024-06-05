from infrence import OpenELMChain
from rouge_score import rouge_scorer
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Evaluate the model")
parser.add_argument(
    "--model", type=str, help="model", default="apple/OpenELM-1_1B-Instruct"
)
parser.add_argument("--lora", type=str, help="LORA output name", default="lora_all_default")
parser.add_argument("--round", type=int, help="Evaluation rounds", default=20)
parser.add_argument(
    "--q4", type=bool, help="Use bnbq4", default=False
)
args = parser.parse_args()

if __name__ == "__main__":
    prompt = """<s>[INST]<SYS>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information.<SYS>{abstract}[/INST]"""
    chain = OpenELMChain(
        prompt=prompt,
        model=args.model,
        lora=f"output/{args.lora}",
        q4=args.q4
    )

    eval_count = args.round

    eval_dataset = pd.read_csv("data/eval.csv")
    eval_dataset = eval_dataset.head(eval_count)
    eval_dataset = Dataset.from_pandas(eval_dataset)

    predicted = chain.invoke([{"abstract": data["abstract"] for data in eval_dataset}])
    eval_scores = []

    for data in tqdm(eval_dataset):
        predicted: str = chain.invoke({"abstract": data["abstract"]})
        predicted = predicted.split("[/INST]")[1]
        reference = data["methods"]

        scorer = rouge_scorer.RougeScorer(["rougeL"])
        eval_scores.append(scorer.score(reference, predicted)["rougeL"].fmeasure)

    print(sum(eval_scores) / eval_count)
