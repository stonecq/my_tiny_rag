from dataclasses import dataclass
from enum import Enum


@dataclass
class ModelBaseConfig:
    model_name:str
    model_dir:str
    device:str='cuda'
###########################################################################################################
################################################ Reranker #################################################
@dataclass
class RerankModelConfig(ModelBaseConfig):
    pass

class RerankModelEnum(Enum):
    BGE_RERANKER_BASE = RerankModelConfig('bge_reranker_base', '/llama/models/bge_reranker_base', 'cuda')



###########################################################################################################
################################################ Embedding Model ##########################################
@dataclass
class EmbeddingModelConfig(ModelBaseConfig):
    pass

class EmbeddingModelEnum(Enum):
    BEG_BASE_ZH_V1_5 = EmbeddingModelConfig('bge_base_zh_v1.5', '/llama/models/bge_reranker_base', 'cuda')

###########################################################################################################
###################################################### LLM ################################################
@dataclass
class LanguageModelConfig(ModelBaseConfig):
    model_type:str='default_type'
class LanguageModelEnum(Enum):
    QWEN_2_5_7B_INSTRUCT = LanguageModelConfig('qwen2.5_7B_Instruct', '/llama/models/qwen2.5_7B_Instruct', 'cuda', 'qwen')

###########################################################################################################
############################################ Sentence Spilt Model #########################################
@dataclass
class SentenceSplitModelConfig(ModelBaseConfig):
    pass
class SentenceSplitModelEnum(Enum):
    NLP_BERT_DOCUMENT_SEGMENTATION_CHINESE_BASE = SentenceSplitModelConfig(
        "nlp_bert_document_segmentation_chinese_base",
        '/llama/models/nlp_bert_document_segmentation_chinese_base',
        'cuda')