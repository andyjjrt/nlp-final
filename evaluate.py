import argparse, logging

logging.basicConfig(level=logging.CRITICAL)

parser = argparse.ArgumentParser(description="Evaluate the model")
parser.add_argument(
    "--model",
    type=str,
    help="Model",
    choices=["270M", "450M", "1_1B", "3B"],
    default="270M",
)
parser.add_argument("--lora", type=str, help="LORA output name")
parser.add_argument("--round", type=int, help="Evaluation rounds", default=100)
parser.add_argument("--q4", help="Use bnbq4", action="store_true")
args = parser.parse_args()

print(args)

from infrence import OpenELMChain
from rouge_score import rouge_scorer
import pandas as pd

if __name__ == "__main__":
    prompt = """<s>[INST]<SYS>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information.<SYS>{abstract}[/INST]"""
    chain = OpenELMChain(
        prompt=prompt,
        model=f"apple/OpenELM-{args.model}-Instruct",
        lora=f"output/{args.lora}" if args.lora else None,
        q4=args.q4,
    )

    eval_count = args.round

    eval_dataset = pd.read_csv("data/test_2.csv")
    eval_dataset = eval_dataset.head(eval_count)

    scorer = rouge_scorer.RougeScorer(["rougeL"])

    predicts = chain.chain.batch(inputs=[d for d in eval_dataset["abstract"]])
    predicts = [p.split("[/INST]")[1] for p in predicts]
    refrences = [d for d in eval_dataset["methods"]]

    eval_scores = [
        scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refrences, predicts)
    ]
    print(sum(eval_scores) / eval_count)
