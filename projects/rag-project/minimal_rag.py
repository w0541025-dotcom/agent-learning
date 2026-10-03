"""Day 2 练习：从零手写最小 RAG。

不依赖 LangChain，四步流程每一环都看得见：
  ① 切分 → ② 嵌入建索引 → ③ 检索 → ④ 拼接生成
运行：python projects/rag-project/minimal_rag.py
"""

import os

# Day 1 排错课的成果：本机代理软件可能干扰国内 API，先清掉代理环境变量
for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)

import numpy as np
from dotenv import load_dotenv
from modelscope import snapshot_download
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# ========== 一个小型知识库：5 段主题互不相同的资料 ==========
DOCUMENTS = [
    "RAG（检索增强生成）是一种给大模型开卷考试的技术：先把私人文档切成片段存进向量库，"
    "用户提问时检索出最相关的片段，连同问题一起交给模型回答。它能减少幻觉、补充模型训练时没有的知识。",

    "文本切分（Chunking）是把长文档切成小块的过程，切分时主要调两个参数：chunk_size（每块多大）"
    "和 chunk_overlap（相邻块重叠多少）。块太大容易混杂多个主题，块太小则语义不完整，经验值是每块 200 到 500 字。",

    "向量嵌入（Embedding）是把一段文字变成一串数字向量（比如 512 维）的技术。语义相近的文字，"
    "向量在空间中的距离也相近，因此可以用余弦相似度衡量两段文字有多像。",

    "混合检索（Hybrid Search）把 BM25 关键词检索和向量语义检索的结果合并，兼顾精确匹配和语义理解，"
    "之后还可以用 Rerank 模型对结果重新排序，进一步提高命中率。",

    "RAG 系统的评估常用 RAGAS 框架，核心指标包括忠实度（回答是否来自检索资料）和答案相关性"
    "（回答是否切题）。评估需要准备一组带标准答案的测试问题。",
]

TOP_K = 2  # 检索时返回最相关的 2 个片段


def split_into_chunks(documents):
    """① 切分：本例的文档每段已是合适大小，直接做清洗；
    真实场景要对长文按 chunk_size/overlap 切割。"""
    return [d.strip() for d in documents if d.strip()]


def build_index(model, chunks):
    """② 嵌入（索引阶段，离线做一次）：每段文字变成一个向量，并归一化。"""
    vectors = model.encode(chunks, normalize_embeddings=True)
    return np.array(vectors)


def retrieve(index, chunks, question_vector, top_k):
    """③ 检索（在线阶段）：向量和问题向量都已归一化，点积就等于余弦相似度。"""
    scores = index @ question_vector
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def generate(context, question):
    """④ 生成：资料+问题一起交给模型；明确要求"不知道就说不知道"，压制幻觉。"""
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL") or None,
    )
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        # 注意：kimi-k2 系列模型只允许 temperature=1，传其他值会报 400
        messages=[
            {"role": "system", "content": "你是严谨的问答助手，只根据给定资料回答。"
                                          "资料里没有答案时直接说不知道，不要编造。"},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("【②】加载本地 embedding 模型（bge-small-zh-v1.5）...")
    model = SentenceTransformer(snapshot_download("BAAI/bge-small-zh-v1.5"))

    chunks = split_into_chunks(DOCUMENTS)
    print(f"【①】知识库切分为 {len(chunks)} 个片段")
    index = build_index(model, chunks)
    print(f"    每个片段已变成一个 {index.shape[1]} 维向量，共 {index.shape[0]} 个\n")

    question = "什么是向量嵌入？"
    print(f"【③】检索问题：{question}")
    question_vector = model.encode([question], normalize_embeddings=True)[0]
    retrieved = retrieve(index, chunks, question_vector, TOP_K)
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"    命中{i}（相似度 {score:.3f}）: {chunk[:36]}...\n")

    context = "\n\n".join(chunk for chunk, _ in retrieved)
    print("【④】把命中片段 + 问题交给 kimi-k2.6 生成：\n")
    print(generate(context, question))
