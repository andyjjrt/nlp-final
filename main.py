from infrence import OpenELM
from rouge_score import rouge_scorer
import argparse

parser = argparse.ArgumentParser(description="Evaluate the model")
parser.add_argument(
    "--model",
    type=str,
    help="Model",
    choices=["270M", "450M", "1_1B", "3B"],
    default="270M",
)
parser.add_argument("--lora", type=str, help="LORA output name")
parser.add_argument("--q4", help="Use bnbq4", action="store_true")
parser.add_argument("--use_config", help="Use generate config", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    abstract = """The reliability of self-labeled data is an important issue when the data are regarded as ground-truth for training and testing learning-based models.
    This paper addresses the issue of false-alarm hashtags in the self-labeled data for irony detection.
    We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection.
    Furthermore, we apply our model to prune the self-labeled training data.
    Experimental results show that the irony detection model trained on the less but cleaner training instances outperforms the models trained on all data."""

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

    predicted: str = model.generate(abstract, generation_config=generation_config)
    print(predicted)

    scorer = rouge_scorer.RougeScorer(["rougeL"])

    reference = """We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection. Furthermore, we apply our model to prune the self-labeled training data."""

    print(scorer.score(reference, predicted)["rougeL"].fmeasure)
