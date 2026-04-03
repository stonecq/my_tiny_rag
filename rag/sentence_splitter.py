import json
from typing import List, Dict, Tuple
from my_tiny_rag.config.model_config.model import SentenceSplitModelConfig
from my_tiny_rag.rag.searcher.doc import Doc


class SplitChunk:
    def __init__(self, text:str, title:str, doc_id:int, chunk_id:int, start_id:int=0,context_ids:List[int]=None):
        self.text = text
        self.title = title
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.context_ids = context_ids
        self.start_id = start_id

    @classmethod
    def from_dict(cls, split_chunk_dict:Dict):
        return cls(
            text=split_chunk_dict.get('text'),
            title=split_chunk_dict.get('title'),
            doc_id=split_chunk_dict.get('doc_id'),
            chunk_id=split_chunk_dict.get('chunk_id'),
            start_id=split_chunk_dict.get('start_id', 0),  # 提供默认值
            context_ids=split_chunk_dict.get('context_ids')  # 如果不存在会返回 None，__init__ 会处理
        )

    def set_context_ids(self, context_ids:List[int]):
        self.context_ids = context_ids

    def get_full_text(self):
        return f'标题:{self.title}。内容:{self.text}'

    def update_chunk_id(self, start_id:int):
        self.chunk_id = self.chunk_id + start_id - self.start_id
        self.context_ids = [context_id + start_id - self.start_id for context_id in self.context_ids]
        self.start_id = start_id

    def to_json(self, ensure_ascii=False, indent=None):
        data = self.__dict__.copy()
        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)


def split_by_priority(doc:Doc, start_id, sentence_size=512, context_len=2):
    title = doc.title
    doc_id = doc.doc_id
    delimiters = [
        "\n\n", "\n \n",
        "\n",
        "。",
        "，", ',',
    ]
    result = []
    def split_by_recursion(split_text:str, split_deep:int):
        if split_deep >= len(delimiters):
            if len(split_text) > sentence_size:
                # 步长为 sentence_size，进行硬切
                for i in range(0, len(split_text), sentence_size):
                    chunk_text = split_text[i: i + sentence_size]
                    if chunk_text.strip():
                        result.append(SplitChunk(chunk_text.strip(), title, doc_id, start_id + len(result), start_id=start_id))
            else:
                # 长度合适了，加入结果
                if split_text.strip():
                    result.append(SplitChunk(split_text.strip(), title, doc_id, start_id + len(result), start_id=start_id))
            return

        parts = split_text.split(delimiters[split_deep])
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= sentence_size:
                result.append(SplitChunk(part, title, doc_id, start_id+len(result), start_id=start_id))
            else:
                split_by_recursion(part, split_deep+1)
    split_by_recursion(doc.text, 0)

    for i in range(len(result)):
        cur_id = start_id + i
        start_context_id = max(0, cur_id-context_len)
        end_context_id = min(len(result), cur_id+context_len)
        result[i].set_context_ids([start_context_id, end_context_id+1])

    return result


class SentenceSplitter:
    def __init__(self, use_model=False,sentence_size=512, config:SentenceSplitModelConfig=None):
        self.sentence_size = sentence_size
        self.use_model = use_model if config is not None else False
        self.config = config

    def split_text(self, doc:Doc) -> List[SplitChunk]:
        return self.split_text_with_id(doc, 0)[0]

    def split_text_with_id(self, doc:Doc, start_id:int) -> Tuple[List[SplitChunk], int]:
        chunk_list = []
        if self.use_model:
            pass
            # result = self.sent_split_pp(documents=sentence)
            # sent_list = [i for i in result["text"].split("\n\t") if i]i
        else:
            chunk_list.extend(split_by_priority(doc, start_id))
        return chunk_list, start_id + len(chunk_list)