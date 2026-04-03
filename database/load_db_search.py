from my_tiny_rag.config.model_config.rag_config import RagConfigEnum
from my_tiny_rag.rag.searcher.searcher import Searcher

db_dir = "/llama/work_space/my_llms_learn/my_tiny_rag/database/wiki"
rag_config = RagConfigEnum.my_tiny_rag_v1.value



if __name__ == "__main__":
    embedding_model_config = rag_config.embedding_model
    ranker_config = rag_config.reranker
    search = Searcher(db_dir, embedding_model_config, ranker_config)
    search.load_db()
    query = "情感计算是什么。"

    result = search.search_with_context(query, top_n=5)
    for score,text in result:
        print(f"分数：{score}，文本：{text}\n")
        print("#"*10)

    # result = search.search(query, top_n=5)
    # for score,chunk in result:
    #     print(f"分数：{score}，文本：{chunk.get_full_text()}\n")
    #     print("#"*10)