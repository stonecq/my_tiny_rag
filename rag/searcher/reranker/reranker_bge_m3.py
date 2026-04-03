import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple

from my_tiny_rag.config.model_config.model import RerankModelConfig
from my_tiny_rag.rag.searcher.reranker.ranker_base import RankerBase
from my_tiny_rag.rag.sentence_splitter import SplitChunk


class RerankerBGEM3(RankerBase):
    def __init__(self, reranker_config: RerankModelConfig, is_api=False) -> None:
        model_path = reranker_config.model_dir
        device = reranker_config.device
        super().__init__(model_path, is_api)

        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def rank(self, query: str, candidate_doc:List[SplitChunk], top_n=3) -> List[Tuple[float, SplitChunk]]:
        # 创建查询和文本对
        pairs = [[query, chunk.get_full_text()] for chunk in candidate_doc]

        # 计算得分
        with torch.no_grad():  # 不计算梯度以节省内存
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(
                self.device)
            outputs = self.model(**inputs, return_dict=True)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        # 将得分和文本对结合，并按得分排序
        scored_query_list = list(zip(scores, candidate_doc))
        scored_query_list.sort(key=lambda x: x[0], reverse=True)  # 按得分降序排列

        # 取前 top_n 的结果
        top_n_results = scored_query_list[:top_n]

        return top_n_results