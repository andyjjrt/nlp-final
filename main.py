from infrence import OpenELMChain
from rouge_score import rouge_scorer
import argparse

parser = argparse.ArgumentParser(description="Evaluate the model")
parser.add_argument(
    "--model", type=str, help="Target model", default="apple/OpenELM-1_1B-Instruct"
)
parser.add_argument("--lora", type=str, help="Output name", default="lora_all_default")
args = parser.parse_args()

if __name__ == "__main__":
    abstract = """The reliability of self-labeled data is an important issue when the data are regarded as ground-truth for training and testing learning-based models.
    This paper addresses the issue of false-alarm hashtags in the self-labeled data for irony detection.
    We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection.
    Furthermore, we apply our model to prune the self-labeled training data.
    Experimental results show that the irony detection model trained on the less but cleaner training instances outperforms the models trained on all data."""

    prompt = """<s>[INST]<SYS>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, no other information.<SYS>{abstract}[/INST]"""

    chain = OpenELMChain(
        prompt=prompt,
        model=args.model,
        lora=f"output/{args.lora}",
    )

    predicted: str = chain.invoke({"abstract": abstract})
    predicted = predicted.split("[/INST]")[1]
    print(predicted)

    scorer = rouge_scorer.RougeScorer(["rougeL"])

    reference = """We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection. Furthermore, we apply our model to prune the self-labeled training data."""

    print(scorer.score(reference, predicted)["rougeL"].fmeasure)
