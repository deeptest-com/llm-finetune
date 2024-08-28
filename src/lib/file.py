import os

def get_project_dir():
    project_dir = os.path.dirname(os.path.abspath(__file__)).split('/files')[0]
    return project_dir

def get_embedding_model_path():
    model_path = '/Users/aaron/.cache/modelscope/hub/AI-ModelScope/bge-large-zh-v1___5'
    return model_path
