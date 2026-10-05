"""Day 4：ANN 索引 benchmark——暴力精确检索 vs Chroma 的 HNSW。

背景：98 个向量时暴力检索毫秒级且精确（Day 3 的 numpy 数组就是暴力检索）；
但向量到十万级，精确检索每次查询都要和全库算一遍距离，扛不住，必须换 ANN。

本实验用 10 万个随机单位向量（ANN 行为主要由向量分布决定，不必真花 embedding 成本）：
1. numpy 暴力精确检索：算出 ground truth（每题真正的 top-10）+ 耗时
2. Chroma（HNSW）：同样的查询 + 耗时
3. 对比 recall@10（Chroma 结果和 ground truth 的重合率）与加速比

运行：python projects/rag-project/ann_benchmark.py
"""

import os
import time
import warnings

for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)

# 本机 macOS 的 BLAS 库对特定矩阵尺寸会误报 RuntimeWarning（Day 3 已排查：
# 数据健康、结果正确），这里压制噪音，用下面的"健康检查"做真正的守卫
warnings.filterwarnings("ignore", category=RuntimeWarning)

import chromadb
import numpy as np
from chromadb.config import Settings

N = 100_000      # 库规模：10 万向量
DIM = 512        # 和 bge 同维度
N_QUERIES = 50   # 测 50 个查询取平均
TOP_K = 10
SEED = 42        # 固定随机种子，结果可复现
BATCH = 5000     # Chroma 单次写入上限 5461，取整 5000
DB_PATH = "projects/rag-project/chroma_bench"


def random_unit_vectors(rng, count):
    v = rng.normal(size=(count, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    library = random_unit_vectors(rng, N)
    queries = random_unit_vectors(rng, N_QUERIES)

    # 健康检查：单位向量的相似度绝不可能超过 1——这是比"警告"更硬的正确性证据
    sanity = float((library @ queries[0]).max())
    assert 0 < sanity <= 1.0, f"相似度越界: {sanity}"
    print(f"健康检查通过：最大相似度 {sanity:.4f}（单位向量必须 ≤1）")

    # ---- 1. 暴力精确检索：ground truth + 耗时 ----
    print(f"库规模 {N} 向量 × {DIM} 维，精确检索建立 ground truth...")
    t0 = time.perf_counter()
    truth = []
    for q in queries:
        top = np.argpartition(-(library @ q), TOP_K)[:TOP_K]
        truth.append(set(top.tolist()))
    exact_ms = (time.perf_counter() - t0) / N_QUERIES * 1000

    # ---- 2. 写入 Chroma（HNSW 索引）----
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection("bench", metadata={"hnsw:space": "cosine"})
    if coll.count() != N:  # 规模不对就重建（改 N 后重跑会自动生效）
        client.delete_collection("bench")
        coll = client.get_or_create_collection("bench", metadata={"hnsw:space": "cosine"})
        t0 = time.perf_counter()
        ids = [f"v{i}" for i in range(N)]
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            coll.add(ids=ids[start:end], embeddings=library[start:end].tolist())
        print(f"写入 {N} 向量（含建 HNSW 索引）耗时 {time.perf_counter() - t0:.1f}s")

    # ---- 3. Chroma 查询：recall@10 + 耗时 ----
    coll.query(query_embeddings=[queries[0].tolist()], n_results=TOP_K)  # 预热一次
    t0 = time.perf_counter()
    recalls = []
    for i, q in enumerate(queries):
        res = coll.query(query_embeddings=[q.tolist()], n_results=TOP_K)
        got = {int(x[1:]) for x in res["ids"][0]}  # "v123" -> 123
        recalls.append(len(got & truth[i]) / TOP_K)
    hnsw_ms = (time.perf_counter() - t0) / N_QUERIES * 1000

    print("\n========== 结果 ==========")
    print(f"暴力精确检索:  {exact_ms:7.2f} ms/查询（召回率 100%，基准线）")
    print(f"Chroma HNSW:   {hnsw_ms:7.2f} ms/查询（召回率 {np.mean(recalls) * 100:.1f}%）")
    print(f"加速比: {exact_ms / hnsw_ms:.1f}x")
    print("\n结论：规模到了 10 万级，HNSW 用一点点召回率换数倍速度；")
    print("而 98 个向量时（Day 3 的文档）暴力检索就已足够——ANN 是为规模准备的。")
