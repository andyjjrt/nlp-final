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
    ```
    usage: dataset.py [-h] [--split_count SPLIT_COUNT] [--start_index START_INDEX]

    Process abstract to methods

    options:
      -h, --help            show this help message and exit
      --split_count SPLIT_COUNT
                            Split range of the file
      --start_index START_INDEX
                            Start index
    ```

2. Merge dataset and split to train/eval split

    ```sh
    python merge_datset.py
    ```
    ```
    Merge data and dump train/val dataset

    options:
      -h, --help   show this help message and exit
      --frac FRAC  Split frac of the file
      --num NUM    Start index
    ```

### Finetune model using LORA

If you have `data.csv`, put it in `data/data.csv`.

```sh
python train.py
```
```
Train model using LORA

options:
  -h, --help            show this help message and exit
  --name NAME           Output name
  --model MODEL         Target model
  --tokenizer TOKENIZER
                        Tokenizer
  --r R                 Lora Config r
  --lora_alpha LORA_ALPHA
                        Lora Config lora_alpha
  --lora_dropout LORA_DROPOUT
                        Lora Config lora_dropout
  --batch_size BATCH_SIZE
                        Batch size, if your vran is low, use 1
  --q4                  Use bnbq4
```

### Evaluate the model

```sh
python evaluate.py
```
```
Evaluate the model

options:
  -h, --help     show this help message and exit
  --model MODEL  model
  --lora LORA    LORA output name
  --round ROUND  Evaluation rounds
  --q4           Use bnbq4
```

### Run sample abstract

```sh
python main.py
```
```
Evaluate the model

options:
  -h, --help     show this help message and exit
  --model MODEL  Target model
  --lora LORA    Output name
  --q4           Use bnbq4
```

#### Sample input

```txt
We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection.
0.8723404255319149
```

## Result

|      |  270M |  1.1B |   3B  |
|------|-------|-------|-------|
| fp32 | 0.747 | 0.842 |  N/A  |
| 4bit | 0.750 | 0.767 | 0.693 |


## Refrence

- LORA config inspired form apple's initial [config](https://github.com/apple/corenet/blob/main/projects/openelm/peft_configs/openelm_lora_1_1B.yaml)
- Trainer options from [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md)