import os
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List
from loguru import logger

from tqdm import tqdm

from my_tiny_rag.config.model_config.rag_config import RagConfig
from my_tiny_rag.rag.llm.qwen_llm import QwenLLM
from my_tiny_rag.rag.searcher.searcher import Searcher
from my_tiny_rag.rag.sentence_splitter import SentenceSplitter
from my_tiny_rag.utils import write_list_to_jsonl

RAG_PROMPT_TEMPLATE = """参考信息:
{context}
---
我的问题或指令：
{question}
---
我的回答:
{answer}
---
请根据上诉参考信息回答和我的问题或指令。前面的参考信息和我的回答可能有用，可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。
回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:"""

def process_docs_text(docs_text: str, sent_split_model) -> List[str]:
    sent_res = sent_split_model.split_text(docs_text)
    return sent_res

class Rag:
    def __init__(self, config: RagConfig):
        self.config = config
        self.searcher = Searcher(
            base_dir=config.base_dir,
            embedding_model_config=config.embedding_model,
            ranker_model_config=config.reranker)
        logger.info('search load')
        if config.language_model.model_type == 'qwen':
            self.llm = QwenLLM(self.config.language_model)
        else:
            raise "failed init LLM, the model type is [qwen2]"

    def build(self, docs:List[str]):
        self.sent_split_model = SentenceSplitter(
            use_model=False,
            config=self.config.sentence_split_model
        )
        split_docs = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_docs_text, item, self.sent_split_model) for item in docs]

        for future in tqdm(as_completed(futures), total=len(docs)):
            try:
                splited = future.result()
                splited = [sent for sent in splited if len(sent)>100]
                split_docs.extend(splited)
            except Exception as e:
                logger.error(f"[RAG][Build]split error {e}")
        self.searcher.build_db(split_docs)
        jsonl_list = [{"text": item} for item in split_docs]
        write_list_to_jsonl(jsonl_list, self.config.base_dir + "/split_sentence.jsonl")
        self.searcher.save_db()

    def load(self):
        self.searcher.load_db()
        logger.info('database load')

    def search(self, query:str, top_n=3) -> str:
        # 第一次回答
        llm_first_resp = self.llm.generate(query)

        # 检索
        concat_query_resp = query + llm_first_resp
        search_result = self.searcher.search_with_context(concat_query_resp, top_n=top_n)
        search_content = [item[1] for item in search_result]
        context = '\n#########################################################\n'.join(search_content)
        # 最终答案
        prompt_text = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
            answer=llm_first_resp
        )
        logger.info(f'prompt:{prompt_text}')
        output = self.llm.generate(prompt_text)
        return output