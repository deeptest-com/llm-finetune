import os
import sys
work_dir = os.getcwd()
sys.path.append(work_dir)

from src.lib.llm import get_embedding_model_path

from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd

from src.config import EmbeddingValDataset, EmbeddingFinetunedModelOutput

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4000"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def evaluate(
        dataset,
        embed_model_path,
        top_k=5,
        verbose=False,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model_path,
        show_progress=True
    )

    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]

        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results


val_dataset = EmbeddingQAFinetuneDataset.from_json(os.path.join(work_dir, "out/val_dataset.json"))

# 原始模型
val_results_origin = evaluate(val_dataset, f"local:{get_embedding_model_path()}")
df_finetuned_origin = pd.DataFrame(val_results_origin)
hit_rate_finetuned = df_finetuned_origin["is_hit"].mean()
print(f"===== before finetuned, hit rate = {hit_rate_finetuned} \n")

# 微调后模型
val_results_finetuned = evaluate(val_dataset, f"local:{EmbeddingFinetunedModelOutput}")
df_finetuned_finetuned = pd.DataFrame(val_results_origin)
hit_rate_finetuned = df_finetuned_finetuned["is_hit"].mean()
print(f"====== after finetuned, hit rate = {hit_rate_finetuned} \n")
