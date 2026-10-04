"""Day 3 实验：chunk_size 对比实验——用数据回答"切太大/太小各有什么坏处"。

方法（这也是工业界构建 RAG 测试集的基本套路）：
1. 从真实文档（教材《强化学习》第一章）划出 5 个"黄金区间"——每道测试题的答案只存在于对应章节
2. 同一份文档分别按 100 / 300 / 800 / 1500 字切分
3. 判分标准：检索 Top-1 片段与黄金区间有重叠 = 命中
4. 对比命中率 + 平均相似度

运行：python projects/rag-project/chunk_experiment.py
"""

import os

for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)

import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

DOC_PATH = "all-in-rag/data/C1/markdown/easy-rl-chapter1.md"

# 测试题 + 黄金区间（用章节标题在文档中的位置划定，答案只会在自己的章节里）
SECTIONS = {
    "q1": ("强化学习和监督学习相比，有哪些假设在强化学习中不成立？", "1.1.1", "1.1.2"),
    "q2": ("强化学习智能体的组成成分有哪些？", "\n## 1.4 ", "\n## 1.5"),
    "q3": ("强化学习中的探索和利用分别指什么？", "\n## 1.6 ", "\n## 1.7"),
    "q4": ("Gym 是什么？在强化学习实验中起什么作用？", "1.7.1", "1.7.2"),
    "q5": ("动作空间分为哪两种？有什么区别？", "\n## 1.3 ", "\n## 1.4"),
}
SIZES = [100, 300, 800, 1500]
OVERLAP_RATIO = 0.2  # 重叠比例：相邻块共享 20%，防止答案被切断


def load_and_clean(path):
    """真实文档很脏：混着图片排版代码，先清洗再切分。"""
    raw = open(path, encoding="utf-8").read()
    lines = [l for l in raw.splitlines() if "<div" not in l and "<img" not in l and "![" not in l]
    return raw, "\n".join(lines)


def chunk_text(text, chunk_size, overlap):
    """固定字符数切分（教学版）。真实项目用段落感知切分（如教材的 Recursive*）。

    返回 [(起始偏移, 块文本)]；纯空白块直接跳过——空块算 embedding 会产生 NaN，
    而 NaN 会让 np.argmax 返回错误结果，悄悄污染整个实验。"""
    step = chunk_size - overlap
    return [(i, text[i:i + chunk_size]) for i in range(0, len(text), step)
            if text[i:i + chunk_size].strip()]


def gold_range(cleaned, start_marker, end_marker):
    """根据章节标题找到黄金区间的字符位置。"""
    gs = cleaned.find(start_marker)
    ge = cleaned.find(end_marker, gs + 1)
    return gs, ge if ge != -1 else len(cleaned)


if __name__ == "__main__":
    raw, text = load_and_clean(DOC_PATH)
    print(f"清洗前 {len(raw)} 字符，清洗后 {len(text)} 字符（丢掉的都是图片排版噪音）\n")

    print("加载 embedding 模型...")
    model = SentenceTransformer(snapshot_download("BAAI/bge-small-zh-v1.5"))

    # 预计算每道题的问题向量和黄金区间
    tests = []
    for q, s_marker, e_marker in SECTIONS.values():
        qv = model.encode([q], normalize_embeddings=True)[0]
        gs, ge = gold_range(text, s_marker, e_marker)
        assert gs != -1, f"找不到章节标记: {s_marker}"
        tests.append((q, qv, gs, ge))

    print(f"\n{'chunk_size':>10} {'块数':>5} {'命中率':>7} {'平均Top1相似度':>14}  各题命中")
    for size in SIZES:
        overlap = int(size * OVERLAP_RATIO)
        chunks = chunk_text(text, size, overlap)                 # [(偏移, 块文本), ...]
        vecs = np.array(model.encode([c for _, c in chunks], normalize_embeddings=True))
        assert not np.isnan(vecs).any(), "存在 NaN 向量，检查是否有空块漏网"

        marks, scores = [], []
        for q, qv, gs, ge in tests:
            top = int(np.argmax(vecs @ qv))        # Top-1 命中的块下标
            cs = chunks[top][0]                    # 该块在原文中的真实起始偏移
            ce = cs + len(chunks[top][1])
            hit = cs < ge and ce > gs              # 与黄金区间有重叠 = 命中
            marks.append("✓" if hit else "✗")
            scores.append(float(vecs[top] @ qv))

        print(f"{size:>10} {len(chunks):>5} {marks.count('✓')}/5{'':>2} "
              f"{np.mean(scores):>12.3f}    {' '.join(marks)}")

    print("\n判读：块太小→语义被切断（✗ 变多）；块太大→主题混杂、相似度被稀释（平均相似度下降）")
