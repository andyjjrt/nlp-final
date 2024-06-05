from typing import Any, Dict, Union
from langchain_core.runnables import RunnableSerializable
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextGenerationPipeline,
    BitsAndBytesConfig
)
from transformers.pipelines import PIPELINE_REGISTRY
from peft import PeftModel
from dotenv import load_dotenv

import os, torch

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


class OpenELMPipeLine(TextGenerationPipeline):
    def preprocess(
        self,
        prompt_text,
        **generate_kwargs,
    ):
        tokenized_prompt = self.tokenizer(prompt_text)
        tokenized_prompt = torch.tensor(tokenized_prompt["input_ids"], device="cuda:0")
        tokenized_prompt = tokenized_prompt.unsqueeze(0)

        return {"input_ids": tokenized_prompt, "prompt_text": prompt_text}


PIPELINE_REGISTRY.register_pipeline(
    "text-generation",
    pipeline_class=OpenELMPipeLine,
    pt_model=AutoModelForCausalLM,
)


class OpenELMChain:
    hf: BaseLLM
    prompt: PromptTemplate
    chain: RunnableSerializable

    def __init__(
        self,
        prompt: str,
        model: str,
        tokenizer: Union[str, AutoTokenizer] = "meta-llama/Llama-2-7b-hf",
        lora: str = None,
        q4: bool = False,
        model_kwargs: Dict[str, Any] = None,
    ):
        self.prompt = PromptTemplate.from_template(prompt)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            token=HF_TOKEN,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
            ) if q4 else None,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            pad_token_id=0,
            model_kwargs=model_kwargs,
        )
        if lora:
            pipe.model = PeftModel.from_pretrained(model, model_id=lora)
        self.hf = HuggingFacePipeline(pipeline=pipe)
        self.chain = self.prompt | self.hf | StrOutputParser()

    def setPrompt(self, prompt: str):
        self.prompt = PromptTemplate.from_template(prompt)
        self.chain = self.prompt | self.hf | StrOutputParser()

    def invoke(self, query: any):
        res = self.chain.invoke(query)
        return res

    def batch(self, query: any):
        res = self.chain.batch(query)
        return res
