import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any


class RankerBase(ABC):
    def __init__(self, model_path: str, is_api: bool = False) -> None:
        super().__init__()
        self.model_path = model_path
        self.is_api = is_api

    @abstractmethod
    def rank(self, query: str, candidate_query: List[str], top_n=3) -> List[Tuple[float, str]]:
        raise NotImplementedError