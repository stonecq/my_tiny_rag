import math
import os.path
import pickle
from typing import List, Tuple, Dict

from tqdm import tqdm

from my_tiny_rag.config.model_config.model import EmbeddingModelConfig, RerankModelConfig
from my_tiny_rag.rag.embedding.hf_embedding import HFSTEmbedding
from my_tiny_rag.rag.searcher.bm25_recall.bm25_retriever import BM25Retriever
from my_tiny_rag.rag.searcher.emb_recall.emb_retriever import EmbRetriever
from my_tiny_rag.rag.searcher.reranker.reranker_bge_m3 import RerankerBGEM3
from my_tiny_rag.rag.sentence_splitter import SplitChunk


class Searcher:
    def __init__(self, base_dir: str, embedding_model_config:EmbeddingModelConfig
                 , ranker_model_config:RerankModelConfig)-> None:
        self.base_dir = base_dir
        self.bm25_retriever = BM25Retriever(base_dir=self.base_dir+"/bm_corpus")
        self.emb_model = HFSTEmbedding(embedding_model_config)
        index_dim = len(self.emb_model.get_embedding(["test_dim"])[0])
        self.emb_retriever = EmbRetriever(index_dim=index_dim, base_dir=self.base_dir + "/faiss_idx")

        self.ranker = RerankerBGEM3(ranker_model_config)
        self.id_to_data : Dict[int, SplitChunk]
        self.id_to_data = dict()


    def build_db(self, docs:List[SplitChunk], batch_size=64):
        self.id_to_data = {chunk.chunk_id:chunk for chunk in docs}
        # 构建词频db
        self.bm25_retriever.build(docs)
        # 构建 embedding db
        total_docs = len(docs)
        num_batches = math.ceil(total_docs / batch_size)

        for i in tqdm(range(num_batches), desc='emb build'):
            start_idx = batch_size * i
            end_idx = min((i + 1) * batch_size, total_docs)
            batch_docs = docs[start_idx:end_idx]
            batch_docs_text = [doc.get_full_text() for doc in batch_docs]
            doc_emb = self.emb_model.get_embedding(batch_docs_text)
            self.emb_retriever.insert_batch(doc_emb, batch_docs)

    def save_db(self):
        self.bm25_retriever.save_bm25_data()
        self.emb_retriever.save()
        with open(os.path.join(self.base_dir, 'id2data.pkl'), 'wb') as f:
            pickle.dump(self.id_to_data, f)

    def load_db(self):
        self.bm25_retriever.load_bm25_data()
        self.emb_retriever.load()
        with open(os.path.join(self.base_dir, 'id2data.pkl'), 'rb') as f:
            self.id_to_data = pickle.load(f)


    def search(self, query:str, top_n=3) -> List[Tuple[float, SplitChunk]]:
        bm25_recall_list = self.bm25_retriever.search(query, 2*top_n)
        query_emb = self.emb_model.get_embedding(query)
        emb_recall_list = self.emb_retriever.search(query_emb, 2*top_n)
        unique_chunk_ids = set()
        unique_chunks = []
        for idx, chunk_id, score in bm25_recall_list:
            if chunk_id not in unique_chunk_ids:
                unique_chunk_ids.add(chunk_id)
                unique_chunks.append(self.id_to_data[chunk_id])

        for chunk_id, score in emb_recall_list:
            if chunk_id not in unique_chunk_ids:
                unique_chunk_ids.add(chunk_id)
                unique_chunks.append(self.id_to_data[chunk_id])
        rerank_result = self.ranker.rank(query, unique_chunks, top_n)

        return rerank_result

    def search_with_context(self, query:str, top_n=3) -> List[Tuple[float, str]]:
        result = self.search(query,top_n)
        result.sort(key=lambda x: x[1].context_ids[0])
        merged = []
        for score, current in result:
            if not merged or current.context_ids[0] > merged[-1][1] + 1:
                merged.append([current.context_ids[0], current.context_ids[1], score])
            else:
                merged[-1][1] = max(merged[-1][1], current.context_ids[1])
                merged[-1][2] = max(merged[-1][2], score)

        final_results = []
        for start_id, end_id, final_score in merged:
            context_texts = []
            for i in range(start_id, end_id + 1):
                if i in self.id_to_data.keys():  # 确保 ID 存在
                    if i == start_id:
                        context_texts.append(self.id_to_data[i].get_full_text())
                    else:
                        context_texts.append(self.id_to_data[i].text)
            full_text = " ".join(context_texts)
            final_results.append((final_score, full_text))
        final_results.sort(key=lambda x: x[0], reverse=True)
        return final_results[:top_n]