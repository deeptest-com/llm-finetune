from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from src.config import EmbeddingTrainDataset, EmbeddingValDataset, EmbeddingValFile, EmbeddingTrainFile
from src.lib.file import traverse_files

from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.finetuning import generate_qa_embedding_pairs

TRAIN_FILES = traverse_files(EmbeddingTrainFile, num=2)
VAL_FILES = traverse_files(EmbeddingValFile, num=2)

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

model = "llama3_cn"
api_key = 'ollama'
# openai_url='http://localhost:11434/v1/'

llm = Ollama(
        model=model,
        api_key=api_key,
        request_timeout=120.0,
        query_wrapper_prompt=PromptTemplate("""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    
                    {query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
                """),
        max_new_tokens=3000,
        context_window=8*1024,
        # openai_url=openai_url,
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