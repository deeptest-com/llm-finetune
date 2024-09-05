import json
import os
import shutil
import sys
import time

from src.lib.file import traverse_files

work_dir = os.getcwd()
sys.path.append(work_dir)
def merge(dir):
    path = os.path.join("out", dir)

    files = traverse_files(path)

    queries_dict = {}
    corpus_dict = {}
    relevant_docs_dict = {}

    for file in files:
        with open(file, 'r') as f:
            if file.endswith(".DS_Store"):
                continue

            try:
                data = json.load(f)
            except Exception as ex:
                print(f"load file {file} error, {ex}")
                continue

        queries = data['queries']
        corpus = data['corpus']
        relevant_docs = data['relevant_docs']

        queries_dict = {**queries_dict, **queries}
        corpus_dict = {**corpus_dict, **corpus}
        relevant_docs_dict = {**relevant_docs_dict, **relevant_docs}

    merged_dict = {
        "queries": queries_dict,
        "corpus": corpus_dict,
        "relevant_docs": relevant_docs_dict,
        "mode": "text"
    }

    with open(path+".json", "w") as outfile:
        json.dump(merged_dict, outfile, indent=4)

merge("train_dataset")

merge("val_dataset")