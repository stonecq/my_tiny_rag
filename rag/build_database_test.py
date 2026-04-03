import logging
import os
import random
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

from tqdm import tqdm

from my_tiny_rag.config.data_config.data_config import TextDatasetEnum
from my_tiny_rag.config.model_config.rag_config import RagConfigEnum
from my_tiny_rag.rag.searcher.doc import Doc
from my_tiny_rag.rag.searcher.searcher import Searcher
from my_tiny_rag.rag.sentence_splitter import SentenceSplitter, SplitChunk
from my_tiny_rag.utils import read_jsonl_to_list, write_list_to_jsonl
######### config #######
########################
rag_config = RagConfigEnum.my_tiny_rag_v1.value
dataset_config = TextDatasetEnum.WIKI_BAIKE.value
output_dir = rag_config.base_dir

data_path = dataset_config.data_path
embedding_model_config= rag_config.embedding_model
reranker_config = rag_config.reranker
sentence_split_model_config = rag_config.sentence_split_model
num_data = 50000

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
if __name__ == "__main__":
    logger.info(
        "启动配置概览:\n"
        "├─ 数据集      : {dataset:<25}\n"
        "├─ 嵌入模型    : {emb:<25}\n"
        "├─ 重排序模型  : {rerank:<25}\n"
        "└─ 分句模型    : {split:<25}".format(
            dataset=dataset_config.data_name,
            emb=embedding_model_config.model_name,
            rerank=reranker_config.model_name,
            split=sentence_split_model_config.model_name
        )
    )
    raw_data_list = read_jsonl_to_list(data_path)
    raw_data_part = random.sample(raw_data_list, num_data)
    docs = []
    for item in raw_data_part:
        title = item.get("title")
        text = item.get("text")
        tags = item.get('tags')
        doc_id = item.get("id")
        docs.append(Doc(title=title, tags=tags, text=text, doc_id=doc_id))
    logger.info(f"数据示例：{raw_data_list[0]}")
    sentence_split_model = SentenceSplitter(use_model=False)

    final_results = [None] * len(docs)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_index = {
            executor.submit(sentence_split_model.split_text, doc): i
            for i, doc in enumerate(docs)
        }
        for future in tqdm(as_completed(future_to_index), total=len(docs)):
            original_index = future_to_index[future]
            try:
                chunks = future.result()
                final_results[original_index] = chunks
            except Exception as exc:
                logger.error(f"Generated an exception: {exc}")

    # 更新id
    result = []
    for doc_chunks in final_results:
        for chunk in doc_chunks:
            chunk.update_chunk_id(len(result))
        result.extend(doc_chunks)

    jsonl_list = [item.to_json() for item in result]
    # write_list_to_jsonl(jsonl_list, os.path.join(output_dir, 'split_sentence.jsonl'))

    logger.info(f"split sentence success, all sentence number: {len(result)}")

    searcher = Searcher(output_dir, embedding_model_config, reranker_config)
    logger.info("load search model success!")
    logger.info("build database ...... ")
    searcher.build_db(result)
    logger.info("build database success, starting save .... ")
    searcher.save_db()
    logger.info("save database success!  ")
