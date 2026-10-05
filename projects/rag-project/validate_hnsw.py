"""验证实验：Chroma HNSW 的 9.2% 召回率，是 Chroma 有问题，还是高维数据的真相？

思路：Chroma 会自报每个返回结果的 distance（cosine 距离 = 1 - 余弦）。
1. 取 chroma 返回的 id，用 numpy 重算这些 id 的真实余弦 → 和 chroma 自报值对比。
   一致 = 存储和打分没坏，召回低是"检索真的没找到最像的"。
2. 对比"真实 top-10 的余弦"和"chroma 找到的最佳余弦" → 看差距有多大。
"""

import os
import warnings

for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import chromadb
import numpy as np
from chromadb.config import Settings

from ann_benchmark import DB_PATH, DIM, N, SEED, random_unit_vectors

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    library = random_unit_vectors(rng, N)
    queries = random_unit_vectors(rng, 50)

    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    coll = client.get_collection("bench")

    for qi in [0, 1, 2]:
        q = queries[qi]
        res = coll.query(query_embeddings=[q.tolist()], n_results=10, include=["distances"])
        chroma_ids = [int(x[1:]) for x in res["ids"][0]]
        chroma_cos = [1 - d for d in res["distances"][0]]                    # chroma 自报的余弦
        real_cos = [float(library[i] @ q) for i in chroma_ids]               # 这些 id 的真实余弦
        true_top = np.argsort(-(library @ q))[:5]
        true_cos = [float(library[i] @ q) for i in true_top]

        storage_ok = np.allclose(chroma_cos, real_cos, atol=1e-4)
        print(f"查询{qi}: 存储一致性 {'✓' if storage_ok else '✗'} | "
              f"真实最佳 {true_cos[0]:.4f} vs chroma 最佳 {max(chroma_cos):.4f}")
        print(f"  真实 top5 余弦: {[round(c, 4) for c in true_cos]}")
        print(f"  chroma top5 余弦: {[round(c, 4) for c in chroma_cos[:5]]}")
