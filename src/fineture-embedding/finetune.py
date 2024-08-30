import os
import sys

from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

from src.config import EmbeddingValDataset, EmbeddingTrainDataset, EmbeddingFinetunedModelOutput

work_dir = os.getcwd()
sys.path.append(work_dir)

from src.lib.file import get_embedding_model_path

train_dataset = EmbeddingQAFinetuneDataset.from_json(EmbeddingTrainDataset)
val_dataset = EmbeddingQAFinetuneDataset.from_json(EmbeddingValDataset)

embedding_model_path = get_embedding_model_path()

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id=embedding_model_path,
    model_output_path=EmbeddingFinetunedModelOutput,
    val_dataset=val_dataset,
)

finetune_engine.finetune()

embed_model = finetune_engine.get_finetuned_model()
print(embed_model)
