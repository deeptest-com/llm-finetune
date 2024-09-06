import os

from src.lib.tool import get_dotenv_var


def get_project_dir():
    project_dir = os.getcwd()
    return project_dir

def traverse_files(dir, start=0, num=-1):
    ret = []

    path = get_project_dir()
    path = os.path.join(path, dir)

    index = 0
    count = 0
    brk = False

    for root, dirs, files in os.walk(path):
        for file in files:
            if index < start:
                index += 1
                continue

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
