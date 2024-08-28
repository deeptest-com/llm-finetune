from langchain_community.llms import Ollama

host = "localhost"
port = "11434"
ollama = Ollama(base_url=f"http://{host}:{port}", model="llama3-zh:latest", temperature=0)

stream = ollama.stream(input="介绍下你自己")

for chunk in stream:
        print(chunk, end="", flush=True)
