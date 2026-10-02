"""Day 1 练习：你的第一个大模型 API 调用。

运行前：把 .env.example 复制为 .env 并填入真实 API_KEY。
运行：  python notes/scripts/hello_llm.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取 .env 文件里的配置

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    # DeepSeek/Kimi/通义等填各自的地址；OpenAI 官方在 .env 里留空即可
    base_url=os.getenv("BASE_URL") or None,
)

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[
        {"role": "system", "content": "你是一个耐心的编程老师，回答简短。"},
        {"role": "user", "content": "用一句话向零基础的人解释什么是 RAG"},
    ],
)

print(response.choices[0].message.content)
