import os
import sys
work_dir = os.getcwd()
sys.path.append(work_dir)

from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

from src.config import EmbeddingValDataset, EmbeddingTrainDataset, EmbeddingFinetunedModelOutput
from src.lib.file import get_embedding_model_path, get_project_dir

work_dir = get_project_dir()

train_dataset = EmbeddingQAFinetuneDataset.from_json(os.path.join(work_dir, "out/train_dataset.json"))
val_dataset = EmbeddingQAFinetuneDataset.from_json(os.path.join(work_dir, "out/val_dataset.json"))

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
