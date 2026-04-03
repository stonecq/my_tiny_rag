from my_tiny_rag.config.data_config.data_config import TextDatasetEnum
from my_tiny_rag.config.model_config.rag_config import RagConfigEnum
from my_tiny_rag.rag.rag_sys import Rag
from my_tiny_rag.utils import read_jsonl_to_list

rag_config = RagConfigEnum.my_tiny_rag_v1.value
dataset_config = TextDatasetEnum.WIKI_BAIKE.value

if __name__ == "__main__":
    query = '请你介绍一下情感计算'
    rag_sys = Rag(rag_config)
    # build database
    # data_list = read_jsonl_to_list(dataset_config.data_path)
    # docs_list = [data_item['text'] for data_item in data_list]
    # rag_sys.build(data_list)

    # load data base
    rag_sys.load()
    resp = rag_sys.search(query)
    print(f"回答：{resp}")