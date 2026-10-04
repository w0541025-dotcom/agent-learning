"""Day 3 下半场：真正的向量库（Chroma）。

对比 minimal_rag.py 里的 numpy 数组，Chroma 多出三样东西：
1. 持久化——数据落盘，重启直接加载，不必重新算 embedding
2. 元数据——每个块带标签（所属章节），支持"先过滤、再检索"
3. ANN 索引——向量到十万级以上时，不再靠暴力遍历（chroma 内置 hnsw）

运行：python projects/rag-project/chroma_store.py（连跑两次，观察第二次的加载速度）
"""

import os

for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)

import chromadb
from chromadb.config import Settings
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

from chunk_experiment import DOC_PATH, chunk_text, load_and_clean  # 复用上午的成果

CHROMA_PATH = "projects/rag-project/chroma_db"
CHUNK_SIZE = 300      # 上午实验的中间值；想验证别的结果就改这里重跑
OVERLAP = 60

# 章节标记：给每个块打上"我属于哪一节"的标签
SECTION_MARKERS = [
    ("1.1", "\n## 1.1 "), ("1.2", "\n## 1.2 "), ("1.3", "\n## 1.3 "),
    ("1.4", "\n## 1.4 "), ("1.5", "\n## 1.5 "), ("1.6", "\n## 1.6 "),
    ("1.7", "\n## 1.7 "),
]


def locate_sections(text):
    """找到每个章节标题在文档中的字符位置，按顺序返回 [(offset, 标签)]。"""
    marks = [(text.find(marker), label) for label, marker in SECTION_MARKERS]
    return sorted((pos, label) for pos, label in marks if pos != -1)


def section_of(offset, marks):
    """根据块的起始偏移，判断它属于哪个章节。"""
    label = "未知"
    for pos, sec in marks:
        if pos <= offset:
            label = sec
        else:
            break
    return label


if __name__ == "__main__":
    text = load_and_clean(DOC_PATH)[1]
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    marks = locate_sections(text)
    print(f"文档切分为 {len(chunks)} 块（chunk_size={CHUNK_SIZE}）")

    # chroma 1.x 默认开启匿名上报，关掉它（也是企业里会做的配置）
    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection("rl_chapter1")

    if coll.count() == 0:
        print("向量库为空 → 计算 embedding 并写入磁盘（首次）...")
        model = SentenceTransformer(snapshot_download("BAAI/bge-small-zh-v1.5"))
        ids = [f"chunk-{i}" for i in range(len(chunks))]
        docs = [c for _, c in chunks]
        metas = [{"section": section_of(off, marks), "offset": off} for off, _ in chunks]
        vecs = model.encode(docs, normalize_embeddings=True).tolist()
        coll.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
        print(f"已写入 {coll.count()} 条向量")
    else:
        print(f"向量库已有 {coll.count()} 条数据 → 直接从磁盘加载，零计算 ⚡")

    model = SentenceTransformer(snapshot_download("BAAI/bge-small-zh-v1.5"))
    question = "强化学习中探索和利用的区别是什么？"
    qv = model.encode([question], normalize_embeddings=True).tolist()[0]

    print(f"\n问题：{question}\n--- 全库检索 Top-3 ---")
    res = coll.query(query_embeddings=[qv], n_results=3)
    for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
        sim = 1 - dist  # chroma 的 cosine 距离 = 1 - 余弦相似度
        print(f"  [相似度 {sim:.3f} | 章节 {meta['section']}] {doc[:40]}...")

    print("--- 只搜 1.6 节（元数据过滤）---")
    res = coll.query(query_embeddings=[qv], n_results=2, where={"section": "1.6"})
    for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
        sim = 1 - dist
        print(f"  [相似度 {sim:.3f} | 章节 {meta['section']}] {doc[:40]}...")
