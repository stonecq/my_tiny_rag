from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn.functional as F

class BaseEmbedding(ABC):
    def __init__(self, path:str, is_api:bool):
        self.path = path
        self.is_api = is_api
        self.name = ''

    @abstractmethod
    def get_embedding(self, text:str):
        raise NotImplemented

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        sim = F.cosine_similarity(torch.Tensor(vector1), torch.Tensor(vector2), dim=-1)
        return sim.numpy().tolist()