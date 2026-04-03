import json
import os.path
from typing import List, Tuple
from my_tiny_rag.rag.searcher.emb_recall.emb_index import EmbIndex
from my_tiny_rag.rag.sentence_splitter import SplitChunk


class EmbRetriever:
    def __init__(self, index_dim:int, base_dir) -> None:
        self.index_dim = index_dim
        self.invert_index = EmbIndex(index_dim)
        self.forward_index: List[int]
        self.forward_index = []
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

    def insert_batch(self, emb_list: List[List], doc_list: List[SplitChunk]):
        if len(emb_list) != len(doc_list):
            raise ValueError(f"向量数量 ({len(emb_list)}) 必须与文档数量 ({len(doc_list)}) 一致")
        if len(emb_list) == 0:
            return

        self.invert_index.insert(emb_list)
        self.forward_index.extend([doc.chunk_id for doc in doc_list])

    def insert(self, emb:List, doc:SplitChunk):
        self.invert_index.insert(emb)
        self.forward_index.append(doc.chunk_id)

    def save(self, index_name=''):
        index_name = index_name if index_name != '' else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, index_name)
        if not  os.path.exists(self.index_folder_path):
            os.makedirs(self.index_folder_path, exist_ok=True)
        with open(self.index_folder_path + "/forward_index.txt", "w", encoding="utf8") as f:
            json.dump(self.forward_index, f)
        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")

    def load(self, index_name=""):
        self.index_name = index_name if index_name != "" else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)

        self.invert_index = EmbIndex(self.index_dim)
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            self.forward_index = json.load(f)

    def search(self, embs: list, top_n=5):
        search_res = self.invert_index.search(embs, top_n)
        recall_list = []
        for idx in range(top_n):
            recall_list.append(
                (self.forward_index[search_res[1][0][idx]], search_res[0][0][idx])) # 文档id/相似度分数
        return recall_list