from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from my_tiny_rag.config.model_config.model import LanguageModelConfig
from my_tiny_rag.rag.llm.base_llm import BaseLLM


class QwenLLM(BaseLLM):
    def __init__(self, config:LanguageModelConfig, is_api:bool=False):
        super().__init__(config, is_api)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_dir,
            torch_dtype='auto',
            trust_remote_code=True,
            device_map=self.config.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_dir,
            trust_remote_code=True
        )
        self.llm_config = AutoConfig.from_pretrained(
            self.config.model_dir,
            trust_remote_code=True
        )
        self.model.eval()

    def generate(self, content: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
