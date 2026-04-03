from dataclasses import dataclass
from enum import Enum

from my_tiny_rag.config.model_config.model import RerankModelConfig, EmbeddingModelConfig, LanguageModelConfig, \
    SentenceSplitModelConfig, RerankModelEnum, EmbeddingModelEnum, SentenceSplitModelEnum, LanguageModelEnum


@dataclass
class RagConfig:
    reranker:RerankModelConfig
    embedding_model:EmbeddingModelConfig
    sentence_split_model:SentenceSplitModelConfig
    language_model:LanguageModelConfig
    base_dir:str


class RagConfigEnum(Enum):
    my_tiny_rag_v1 = RagConfig(
        reranker=RerankModelEnum.BGE_RERANKER_BASE.value,
        embedding_model=EmbeddingModelEnum.BEG_BASE_ZH_V1_5.value,
        sentence_split_model=SentenceSplitModelEnum.NLP_BERT_DOCUMENT_SEGMENTATION_CHINESE_BASE.value,
        language_model=LanguageModelEnum.QWEN_2_5_7B_INSTRUCT.value,
        base_dir="/llama/work_space/my_llms_learn/my_tiny_rag/database/wiki"
    )