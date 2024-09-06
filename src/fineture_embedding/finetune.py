import os
import sys

from src.lib.llm import get_embedding_model_path

work_dir = os.getcwd()
sys.path.append(work_dir)

from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

from src.config import EmbeddingValDataset, EmbeddingTrainDataset, EmbeddingFinetunedModelOutput
from src.lib.file import get_project_dir

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
