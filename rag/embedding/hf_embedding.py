from typing import List, Union

import numpy as np

from my_tiny_rag.config.model_config.model import EmbeddingModelConfig
from my_tiny_rag.rag.embedding.base_embedding import BaseEmbedding
from sentence_transformers import SentenceTransformer, util

class HFSTEmbedding(BaseEmbedding):
    def __init__(self, embedding_model_config:EmbeddingModelConfig, is_api:bool=False)-> None:
        path = embedding_model_config.model_dir
        device = embedding_model_config.device
        super().__init__(path, is_api)
        self.st_model = SentenceTransformer(path, device=device)
        self.name = "hf_model"

    def get_embedding(self, texts: Union[str, List[str]]):
        is_single_input = isinstance(texts, str)
        if is_single_input:
            input_data = [texts]
        else:
            input_data = texts
        st_embedding = self.st_model.encode(input_data, convert_to_numpy=True, show_progress_bar=False)
        emb_array = np.array(st_embedding)
        if emb_array.ndim == 1:
            emb_array = emb_array.reshape(1, -1)

        # L2 归一化
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 防止除以0
        normalized_embeddings = emb_array / norms

        result_list = normalized_embeddings.tolist()
        return result_list
