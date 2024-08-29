from langchain_community.llms import Ollama

from src.fineture_llama3.helper import get_alpaca_prompt

host = "localhost"
port = "11434"
ollama = Ollama(base_url=f"http://{host}:{port}", model="llama3-zh:latest", temperature=0)

msg = "介绍下你自己"
msg = get_alpaca_prompt().format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )

stream = ollama.stream(input=msg)

for chunk in stream:
        print(chunk, end="", flush=True)
