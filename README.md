# NLP Final

## Requirement
  - `conda` or `mamba` installed
  - `nvidia` device that has `cuda`
  - `ollama` if you want to prepare dataset yourself

## Install

```sh
conda env create -f environment.yml
```

### Environent variable

```
HF_TOKEN=xxx
OPENAI_API_KEY=xxx
```

## Datasets

The original dataset is sourced from [gfissore/arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021). From this dataset, we randomly selected 16,000 abstracts with lengths between 512 and 2048 characters, ensuring each abstract is a meaningful paragraph.

### Data Generation and Filtering

1. **Initial Abstract Selection**:
    - Selected 16,000 abstracts with lengths between 512 and 2048 characters.

2. **Answer Generation**:
    - Utilized `llama3-4bit` to generate answers based on the selected abstracts.
    - Retained only those instances where the generated answer was a sub-sentence of the original abstract.

3. **Methods Generation**:
    - Selected a small subset (10%) from the filtered data.
    - Used `gpt4o` to generate methods for these abstracts.

### Dataset Composition

We created three primary datasets:

- `1k`: Contains 1,000 samples with 90% `llama3-4bit` generated answers and 10% `gpt4o` generated methods.
- `5k`: Contains 5,000 samples with the same ratio of `llama3-4bit` to `gpt4o` data.
- `10k`: Contains 10,000 samples with the same ratio of `llama3-4bit` to `gpt4o` data.

Additionally, we constructed combined datasets:

- `6k`: Composed of the `1k` and `5k` datasets.
- `16k`: Composed of the `1k`, `5k`, and `10k` datasets.
- `gpt`: Composed of all `gpt4o` generated data, and samples from the `10k` dataset using `chatGPT4`'s answers, totaling about 2.1k data points.

### Validation Dataset

- Contains 100 abstracts different from the training datasets, also adhering to the length limit.
- All abstracts are based on `chatGPT`, but supervised by humans.

### Accessing the Datasets

You can access different branches in [nccu-1122-nlp-final/arxiv-abstracts-methods](https://huggingface.co/datasets/nccu-1122-nlp-final/arxiv-abstracts-methods) for different datasets

## Finetune

This project focuses on finetuning models from the `OpenELM` series using supervised and unsupervised LoRA/QLoRA methods. Key parameters can be adjusted via command line arguments.

### Command Line Usage

```sh
Train model using LORA

options:
  -h, --help            show this help message and exit
  --model {270M,450M,1_1B,3B}
                        Target model
  --tokenizer TOKENIZER
                        Tokenizer
  --dataset {1k,5k,6k,10k,16k, gpt}
                        Dataset size
  --r R                 Lora Config r
  --lora_alpha LORA_ALPHA
                        Lora Config lora_alpha
  --lora_dropout LORA_DROPOUT
                        Lora Config lora_dropout
  --batch_size BATCH_SIZE
                        Batch size, if your vram is low, use 1
  --q4                  Use bnbq4
```

### Key Parameters

- **Model Selection**: Choose from models of varying sizes: 270M, 450M, 1.1B, or 3B parameters.
- **Tokenizer**: Specify the tokenizer to preprocess the data.
- **Dataset Size**: Select dataset size from 1k, 5k, 6k, 10k, 16k, or GPT.
- **LoRA Configuration**:
  - `--r`: Set the rank for LoRA.
  - `--lora_alpha`: Set the alpha value for LoRA.
  - `--lora_dropout`: Set the dropout rate for LoRA.
- **Batch Size**: Adjust according to available VRAM. For low VRAM, set batch size to 1.
- **Quantization**: Use the `--q4` flag to enable bnbq4 quantization.

## Evaluate the Model

For evaluation, we use the average `rougeL` score. Simply pass the adapter name, and it will automatically evaluate. You can also enable generation configuration using the `--use_config` flag.

### Command Line Usage

```sh
Evaluate the model

options:
  -h, --help            show this help message and exit
  --model {270M,450M,1_1B,3B}
                        Model
  --lora LORA           LORA output name
  --round ROUND         Evaluation rounds
  --q4                  Use bnbq4
  --use_config          Use generate config
```

### Key Parameters

- **Model Selection**: Choose from models of varying sizes: 270M, 450M, 1.1B, or 3B parameters.
- **LoRA Output**: Specify the name of the LoRA output to be evaluated.
- **Evaluation Rounds**: Define the number of rounds for evaluation.
- **Quantization**: Use the `--q4` flag to enable bnbq4 quantization.
- **Generation Configuration**: Use the `--use_config` flag to enable generation configuration during evaluation.

## Sample input

```
> python main.py --model 270M --lora=loraV2_sft_270M_16k

We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection. Furthermore, we apply our model to prune the self-labeled training data.
1.0

```

## Result

- V1: llama3 generated methods, no furthur data selection
  - 270M
    - LoRA 2k data: 0.5812870151234503
    - LoRA 5k data: 0.6072088812985723
    - LoRA 10k data: 0.6062697605573064
    - QLoRA 2k data: 0.5344934219662798
    - QLoRA 5k data: 0.5716003366469663
  - 1.1B
    - LoRA 2k data: 0.61272390122016
    - LoRA 10k data: 0.6395614379454786
    - QLoRA 2k data: 0.6095495084388393
- V2: 90% llama3 10% gpt, 512 < abstrat length < 2048
  - 270M
    - LoRA 1k data: 0.5535659518788367
    - LoRA 5k data: 0.6100692579923299
    - LoRA 6k data: 0.6131488741306242
    - LoRA 10k data: 0.5955115766782487
    - LoRA 16k data: 0.6345286218576365
    - LoRA SFT 1k data: 0.560211279893723
    - LoRA SFT 5k data: 0.6109013763000987
    - LoRA SFT 6k data: 0.6038713730981348
    - LoRA SFT 10k data: 0.6141869259514844
    - LoRA SFT 16k data: **0.6434976595372202**
    - LoRA SFT gpt data: 0.5748198221119681
  - 450M
    - LoRA SFT 16k data: 0.6454982837614324


## Refrences

- LORA config inspired form apple's initial [config](https://github.com/apple/corenet/blob/main/projects/openelm/peft_configs/openelm_lora_1_1B.yaml)
- Trainer options from [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md)