import os

from src.lib.tool import get_dotenv_var


def get_project_dir():
    project_dir = os.getcwd()
    return project_dir

def get_embedding_model_path():
    model_path = get_dotenv_var('EmbeddingModulePath')
    return model_path

def get_llama3_model_path():
    model_path = get_dotenv_var('Llama3ModulePath')
    return model_path

def traverse_files(dir, num=3):
    ret = []

    path = get_project_dir()
    path = os.path.join(path, dir)

    count = 0
    brk = False

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)

            ret.append(file_path)

            count += 1
            if 0 < num <= count:
                brk = True
                break

        if brk:
            break

    return ret