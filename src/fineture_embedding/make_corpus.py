import os
import sys
work_dir = os.getcwd()
sys.path.append(work_dir)

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser

from src.config import EmbeddingTrainDataset, EmbeddingValDataset, EmbeddingValFile, EmbeddingTrainFile
from src.lib.file import traverse_files

from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.finetuning import generate_qa_embedding_pairs

TRAIN_FILES = traverse_files(EmbeddingTrainFile)
# VAL_FILES = traverse_files(EmbeddingValFile)

TRAIN_CORPUS_FPATH = EmbeddingTrainDataset
VAL_CORPUS_FPATH = EmbeddingValDataset

def load_corpus(docs, for_training=False, verbose=False):
    # parser = SimpleNodeParser.from_defaults()
    parser = SentenceSplitter()

    num = int(len(docs) * 0.7)

    if for_training:
        nodes = parser.get_nodes_from_documents(docs[:num], show_progress=verbose)
    else:
        nodes = parser.get_nodes_from_documents(docs[num:], show_progress=verbose)

    if verbose:
        print(f'Parsed {len(nodes)} nodes')

    return nodes

reader = SimpleDirectoryReader(input_files=TRAIN_FILES)
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

train_nodes = load_corpus(docs, for_training=True, verbose=True)
val_nodes = load_corpus(docs, for_training=False, verbose=True)

model = "llama3_cn"
base_url='http://vvtg1184983.bohrium.tech:50001'

llm = Ollama(
        model=model,
        request_timeout=600.0,
        query_wrapper_prompt=PromptTemplate("""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    
                    {query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
                """),
        max_new_tokens=3000,
        context_window=8*1024,
        base_url=base_url,
    )

train_dataset = generate_qa_embedding_pairs(llm=llm, nodes=train_nodes)
val_dataset = generate_qa_embedding_pairs(llm=llm, nodes=val_nodes)

train_dataset.save_json(EmbeddingTrainDataset)
val_dataset.save_json(EmbeddingValDataset)

# train_dataset = EmbeddingQAFinetuneDataset.from_json(TrainDataset)
# val_dataset = EmbeddingQAFinetuneDataset.from_json(ValDataset)