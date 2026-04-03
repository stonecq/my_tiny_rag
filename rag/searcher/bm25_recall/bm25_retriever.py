import os.path
import pickle
from typing import List

import jieba
from tqdm import tqdm

from my_tiny_rag.rag.searcher.bm25_recall.rank_bm25 import BM25Okapi
from my_tiny_rag.rag.sentence_splitter import SplitChunk


class BM25Retriever:
    def __init__(self, base_dir:str, docs: List[SplitChunk]=None):
        self.chunk_id_list = docs
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir,exist_ok=True)
        if self.chunk_id_list is not None and len(self.chunk_id_list) != 0:
            self.build(self.chunk_id_list)
        else:
            print("未初始化数据库")

    def tokenize(self, txt:str)-> List[str]:
        return list(jieba.cut_for_search(txt))

    def build(self, docs:List[SplitChunk]):
        self.chunk_id_list = [chunk.chunk_id for chunk in docs]
        self.tokenized_corpus = []
        for doc in tqdm(docs, desc='bm25 build'):
            tokenized = self.tokenize(doc.get_full_text())
            self.tokenized_corpus.append(tokenized)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save_bm25_data(self, db_name=""):
        """ 对数据进行分词并保存到文件中。
        """
        db_name = db_name if db_name != "" else "bm25_data"
        db_file_path = os.path.join(self.base_dir, db_name + ".pkl")
        # 保存分词结果
        data_to_save = {
            "chunk_id_list": self.chunk_id_list,
            "tokenized_corpus": self.tokenized_corpus
        }

        with open(db_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_bm25_data(self, db_name=''):
        db_name=db_name if db_name != '' else "bm25_data"
        db_file_path = os.path.join(self.base_dir, db_name + '.pkl')
        with open(db_file_path, 'rb') as f:
            data = pickle.load(f)
        self.chunk_id_list = data['chunk_id_list']
        self.tokenized_corpus = data['tokenized_corpus']
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query:str, top_n=5):
        if self.tokenized_corpus is None:
            raise ValueError("Tokenized corpus is not loaded or generated.")

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取分数最高的前N个文本索引
        top_n_indices = sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)[:top_n]

        result = [
            (i, self.chunk_id_list[i], scores[i])
            for i in top_n_indices
        ]
        return result