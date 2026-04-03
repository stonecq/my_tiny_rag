import numpy as np
import faiss


class EmbIndex:
    def __init__(self, index_dim: int) -> None:
        # description = "HNSW64"
        # measure = faiss.METRIC_L2
        # self.index = faiss.index_factory(index_dim, description, measure)

        # 数据量小，用暴力搜索算法
        self.index = faiss.IndexFlatL2(index_dim)

    def insert(self, emb: list):
        """
        接收一个或一组向量（嵌入），将其转换为 Faiss 要求的格式，然后存入索引中。
        :param emb: List[float] or List[List[float]]
        """
        emb = np.array(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)
        self.index.add(emb)

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def search(self, vec: list, num: int):
        vec = np.array(vec, dtype=np.float32)  # 转换为 NumPy 数组
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)  # 转换为 (1, d) 形状
        return self.index.search(vec, num)