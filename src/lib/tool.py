import os

from dotenv import load_dotenv, find_dotenv

def get_openai_key():
    _ = load_dotenv(find_dotenv())

    return os.environ['OPENAI_API_KEY_PROXY']


def get_openai_url():
    _ = load_dotenv(find_dotenv())

    return os.environ['OPENAI_API_URL_PROXY']