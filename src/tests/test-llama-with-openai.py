from openai import OpenAI

client = OpenAI(
    base_url='http://vvtg1184983.bohrium.tech:50001/v1/',
    api_key='ollama', # required but ignored
)

stream = client.chat.completions.create(
    model='llama3_cn',
    stream=True,
    messages=[
        {
            "role": "user",
            "content": "你好！"
        },
        {
            "role": "assistant",
            "content": "我是一个数学老师。"
        },
        {
            "role": "user",
            "content": "你在学校教什么?"
        }
    ],
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
