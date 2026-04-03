from abc import ABC, abstractmethod

from my_tiny_rag.config.model_config.model import LanguageModelConfig


class BaseLLM(ABC):

    def __init__(self, language_model_config: LanguageModelConfig, is_api:bool=False):
        super().__init__()
        self.config: LanguageModelConfig
        self.is_api = is_api
        self.config = language_model_config
    @abstractmethod
    def generate(self, content: str) -> str:
        raise NotImplementedError