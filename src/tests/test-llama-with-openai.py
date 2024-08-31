from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama', # required but ignored
)

stream = client.chat.completions.create(
    model='llama3_cn',
    stream=True,
    messages=[
        {
            'role': 'user',
            'content': '介绍下你自己',
        }
    ],
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
