# NLP Final

## Requirement
  - `conda` or `mamba` installed
  - `nvidia` device that has `cuda`
  - `ollama` if you want to prepare dataset yourself

## Install

```sh
mamba env create -f environment.yml
```

## Run

### Prepare dataset

1. Generate dataset

    > This use llama3 to generate dataset, aims to make openELM be as smart as llama3 in abstract summarization, a better choice is using GPT4, but i'm poor.

    ```sh
    python dataset.py
    ```

2. Merge dataset and split to train/eval split

    ```sh
    python merge_datset.py
    ```

### Finetune model using LORA

If you have `data.csv`, put it in `data/data.csv`.

```sh
python train.py
```

### Evaluate the model

```sh
python evaluate.py
```

### Run sample abstract

```sh
python main.py
```

```txt
We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection.
0.8723404255319149
```

## Refrence

- LORA config inspired form apple's initial [config](https://github.com/apple/corenet/blob/main/projects/openelm/peft_configs/openelm_lora_1_1B.yaml)
- Trainer options from [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md)