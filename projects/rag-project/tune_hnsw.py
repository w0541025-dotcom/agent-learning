"""调参实验 v2：建集合时就把 HNSW 参数写死，对比默认参数的基线。

v1 的教训：coll.modify() 改 search_ef 后四个档位结果一模一样——
说明 modify 对索引参数只是"存了个值"，已加载的索引并不重新绑定。
所以 v2 在 get_or_create_collection 的 metadata 里直接指定参数建库。
"""

import os
import time
import warnings

for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import chromadb
import numpy as np
from chromadb.config import Settings

from ann_benchmark import BATCH, DB_PATH, N, N_QUERIES, SEED, TOP_K, random_unit_vectors

PARAMS = {"hnsw:space": "cosine", "hnsw:M": 32, "hnsw:construction_ef": 400, "hnsw:search_ef": 400}

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    library = random_unit_vectors(rng, N)
    queries = random_unit_vectors(rng, N_QUERIES)

    truth = []
    for q in queries:
        top = np.argpartition(-(library @ q), TOP_K)[:TOP_K]
        truth.append(set(top.tolist()))

    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection("bench2", metadata=PARAMS)
    if coll.count() != N:
        client.delete_collection("bench2")
        coll = client.get_or_create_collection("bench2", metadata=PARAMS)
        t0 = time.perf_counter()
        ids = [f"v{i}" for i in range(N)]
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            coll.add(ids=ids[start:end], embeddings=library[start:end].tolist())
        print(f"写入 {N} 向量（{PARAMS}）耗时 {time.perf_counter() - t0:.1f}s")

    coll.query(query_embeddings=[queries[0].tolist()], n_results=TOP_K)
    t0 = time.perf_counter()
    recalls = []
    for i, q in enumerate(queries):
        res = coll.query(query_embeddings=[q.tolist()], n_results=TOP_K)
        got = {int(x[1:]) for x in res["ids"][0]}
        recalls.append(len(got & truth[i]) / TOP_K)
    ms = (time.perf_counter() - t0) / N_QUERIES * 1000

    print("\n========== 对比 ==========")
    print(f"默认参数 (M=16):           1.58 ms/查询, 召回率  9.2%")
    print(f"调参后   (M=32, ef=400):   {ms:.2f} ms/查询, 召回率 {np.mean(recalls) * 100:.1f}%")
