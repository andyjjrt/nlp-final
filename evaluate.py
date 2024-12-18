import argparse, logging
from tqdm import tqdm

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
parser.add_argument("--use_config", help="Use generate config", action="store_true")
parser.add_argument("--debug", help="print predicted and refrence", action="store_true")
args = parser.parse_args()

from infrence import OpenELM
from rouge_score import rouge_scorer
import pandas as pd

if __name__ == "__main__":

    model = OpenELM(
        model=f"apple/OpenELM-{args.model}-Instruct",
        lora=f"output/{args.lora}" if args.lora else None,
        q4=args.q4,
    )

    generation_config = {
        "do_sample": True,
        "temperature": 1,
        "top_k": 0,
    }

    eval_count = args.round
    eval_dataset = pd.read_csv("data/test.csv")
    eval_dataset = eval_dataset.head(eval_count)
    eval_scores = []
    scorer = rouge_scorer.RougeScorer(["rougeL"])

    for index, data in  eval_dataset.iterrows() if args.debug else tqdm(eval_dataset.iterrows(), total=eval_count):
        predicted: str = model.generate(data["abstract"], generate_kwargs=generation_config if args.use_config else None)
        score = scorer.score(data["methods"], predicted)["rougeL"].fmeasure
        eval_scores.append(score)
        if args.debug:
            print("=" * 30)
            print(f"Abstract: {data["abstract"]}")
            print("-" * 30)
            print(f"Method: {data["methods"]}")
            print("-" * 30)
            print(f"Predicted: {predicted}")
            print("-" * 30)
            print(f"Score: {score}")
            print("=" * 30)

    print(f"Average: {sum(eval_scores) / eval_count}")
    print(f"Matched: {len([s for s in eval_scores if s == 1])}/{eval_count}")
    print(f"Score over 0.8: {len([s for s in eval_scores if s > 0.8])}/{eval_count}")
    print(f"Score over 0.5: {len([s for s in eval_scores if s > 0.5])}/{eval_count}")
