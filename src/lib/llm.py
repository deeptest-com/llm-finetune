import os

from src.lib.tool import get_dotenv_var


def get_embedding_model_path():
    model_path = get_dotenv_var('EmbeddingModulePath')
    return model_path

def get_llama3_model_path():
    model_path = get_dotenv_var('Llama3ModulePath')
    return model_path
