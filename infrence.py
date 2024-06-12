from typing import Any, Dict, Optional
from langchain_core.prompts import PromptTemplate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig,
)
from peft import PeftModel
from dotenv import load_dotenv

import os, torch

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_PROMPT = "<s>[INST]<SYS>From the following abstract, extract the sentences that shows the methods of the research. Only the sentences from the abstract, and NOTHING else.<SYS>{abstract}[/INST]"


class OpenELM:
    prompt: PromptTemplate
    tokenizer: AutoTokenizer

    def __init__(
        self,
        model: str,
        prompt: str = DEFAULT_PROMPT,
        tokenizer: str = "meta-llama/Llama-2-7b-hf",
        lora: str = None,
        q4: bool = False,
        model_kwargs: Dict[str, Any] = None,
    ):
        self.prompt = PromptTemplate.from_template(prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            token=HF_TOKEN,
            trust_remote_code=True,
            quantization_config=(
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_storage=torch.bfloat16,
                )
                if q4
                else None
            ),
            device_map="auto",
             **(model_kwargs if model_kwargs else {}),
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if lora:
            self.model = PeftModel.from_pretrained(self.model, model_id=lora)

    def generate(
        self,
        inputs: str,
        generate_kwargs: Optional[dict] = None,
    ):
        inputs = self.prompt.format(abstract=inputs)
        tokenized_inputs = self.tokenizer(inputs)
        tokenized_inputs = torch.tensor(
            tokenized_inputs["input_ids"], device=self.model.device
        )
        tokenized_inputs = tokenized_inputs.unsqueeze(0)
        
        set_seed(0)

        output_ids = self.model.generate(
            tokenized_inputs,
            max_length=1024,
            **(generate_kwargs if generate_kwargs else {}),
        )

        output_text = self.tokenizer.decode(
            output_ids[0][tokenized_inputs.shape[1] :].tolist(),
            skip_special_tokens=True,
        )

        return output_text
