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

## Run

### Prepare dataset

1. Split dataset chunks
    - 1k, 6k, 10k with 5x data

2. Generate dataset
    - llama3
    - gpt4o

3. Merge dataset

### Finetune model using LORA

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

#### Sample input

```txt
We analyze the ambiguity of hashtag usages and propose a novel neural network-based model, which incorporates linguistic information from different aspects, to disambiguate the usage of three hashtags that are widely used to collect the training data for irony detection.
0.8723404255319149
```

## Result


- V1: llama3 generated methods, no furthur data selection
  - 270M
    - LoRA 2k data: 0.4052257085960178
    - LoRA 5k data: 0.45824485894693273
    - LoRA 10k data: 0.4489090268697917
    - QLoRA 2k data: 0.37168843564664866
    - QLoRA 5k data: 0.4211203208519094
    
- V2: 90% llama3 10% gpt, 512 < abstrat length < 2048
  - 270M
    - LoRA 1k data: 0.4000376466453418
    - LoRA 5k data: 0.4396882406322232
    - LoRA 6k data: 0.43338958159823016
    - LoRA 10k data: 0.4339534568103959
    - LoRA 16k data: 0.47119269373354755
  - 450M
    - LoRA 1k data: 0.3838928409670989
    - LoRA 5k data: 0.41486331409663285
    - LoRA 6k data: 0.43090487953131495
    - LoRA 10k data: 0.44012725827382954
    - LoRA 16k data: 0.444286742569119
  - 1.1B
    - LoRA 1k data: 0.3796559238097318
    - LoRA 5k data: 0.4313785272161937
    - LoRA 6k data: 0.4323177185339683
    - LoRA 10k data: 0.4478788758647639
    - LoRA 16k data: 0.4582330700426558



## Refrence

- LORA config inspired form apple's initial [config](https://github.com/apple/corenet/blob/main/projects/openelm/peft_configs/openelm_lora_1_1B.yaml)
- Trainer options from [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md)