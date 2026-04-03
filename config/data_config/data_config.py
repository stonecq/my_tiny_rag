from dataclasses import dataclass
from enum import Enum

text_data_list = [

]
@dataclass
class DataConfig:
    data_name:str
    data_path:str
    data_type:str

class TextDatasetConfig(DataConfig):
    pass

class TextDatasetEnum(Enum):
    WIKI_BAIKE = TextDatasetConfig('wiki_baike', '/llama/datasets/tiny_rag/wikipedia-zh-cn-20250320.json', 'text')