from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from src.config import EmbeddingTrainDataset, EmbeddingValDataset, EmbeddingValFile, EmbeddingTrainFile
from src.lib.tool import get_openai_key, get_openai_url

TRAIN_FILES = [EmbeddingTrainFile]
VAL_FILES = [EmbeddingValFile]

TRAIN_CORPUS_FPATH = EmbeddingTrainDataset
VAL_CORPUS_FPATH = EmbeddingValDataset

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes

train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.llms.openai import OpenAI

model = "gpt-4o"
llm = OpenAI(
        model=model,
        api_key=get_openai_key(),
        api_base=get_openai_url(),
    )

train_dataset = generate_qa_embedding_pairs(
    llm=llm, nodes=train_nodes
)
val_dataset = generate_qa_embedding_pairs(
    llm=llm, nodes=val_nodes
)

train_dataset.save_json(EmbeddingTrainDataset)
val_dataset.save_json(EmbeddingValDataset)

# train_dataset = EmbeddingQAFinetuneDataset.from_json(TrainDataset)
# val_dataset = EmbeddingQAFinetuneDataset.from_json(ValDataset)