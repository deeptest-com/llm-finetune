import ollama

from src.fineture_llama3.helper import get_alpaca_prompt

msg = 'Why is the sky blue?'
msg = get_alpaca_prompt().format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )

stream = ollama.chat(
    model='llama3_cn',
    messages=[{'role': 'user', 'content': msg}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)