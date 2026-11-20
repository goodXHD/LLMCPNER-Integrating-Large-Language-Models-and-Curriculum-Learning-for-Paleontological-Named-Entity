#!/usr/bin/env python3
"""
K-Means聚类后从每个簇提取样本。

用法：
    conda run -n unsloth python extract_cluster_samples.py --k 100 --samples 30
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

import argparse
import json
import random
from collections import Counter
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def load_abstracts(abstracts_file: str) -> List[dict]:
    """加载摘要数据"""
    with open(abstracts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('abstracts', [])


def encode_abstracts(abstracts: List[dict], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """使用 sentence-transformers 将摘要编码为向量"""
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"加载模型: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(model_name, device=device)
    texts = [item['abstract'] for item in abstracts]
    print(f"编码 {len(texts)} 个摘要...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=256
    )
    return embeddings


def cluster_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    """执行 K-Means 聚类"""
    print(f"执行 K-Means 聚类 (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, verbose=1)
    labels = kmeans.fit_predict(embeddings)
    return labels


def extract_cluster_keywords(abstracts: List[dict], labels: np.ndarray, top_n: int = 10) -> dict:
    """提取每个聚类的关键词"""
    print("提取聚类关键词...")
    unique_labels = sorted(set(labels))
    cluster_texts = {label: [] for label in unique_labels}

    for i, label in enumerate(labels):
        cluster_texts[label].append(abstracts[i]['abstract'])

    cluster_keywords = {}
    for cluster_id in unique_labels:
        texts = cluster_texts[cluster_id]
        if len(texts) < 2:
            cluster_keywords[cluster_id] = []
            continue

        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            cluster_keywords[cluster_id] = keywords
        except Exception:
            cluster_keywords[cluster_id] = []

    return cluster_keywords


def extract_samples(abstracts: List[dict], labels: np.ndarray, samples_per_cluster: int) -> dict:
    """从每个簇提取样本"""
    print(f"从每个簇提取最多 {samples_per_cluster} 个样本...")

    unique_labels = sorted(set(labels))
    label_counts = Counter(labels)

    clusters_data = []
    for cluster_id in unique_labels:
        # 获取该簇的所有索引
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_count = len(cluster_indices)

        # 随机抽样
        sample_count = min(samples_per_cluster, cluster_count)
        sampled_indices = random.sample(cluster_indices, sample_count)

        samples = []
        for idx in sampled_indices:
            samples.append({
                'category': abstracts[idx]['category'],
                'filename': abstracts[idx]['filename'],
                'abstract': abstracts[idx]['abstract']
            })

        clusters_data.append({
            'cluster_id': int(cluster_id),
            'count': int(cluster_count),
            'samples': samples
        })

    return {
        'num_clusters': len(unique_labels),
        'samples_per_cluster': samples_per_cluster,
        'total_abstracts': len(abstracts),
        'clusters': clusters_data
    }


def main():
    parser = argparse.ArgumentParser(description='K-Means聚类后提取样本')
    parser.add_argument(
        '--abstracts-file',
        type=str,
        default='data/abstracts.json',
        help='摘要 JSON 文件路径'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='outputs/cluster_samples.json',
        help='输出 JSON 文件路径'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=100,
        help='K-Means 聚类数量'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=30,
        help='每个簇提取的样本数量'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='sentence-transformers 模型名称'
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # 加载摘要
    print(f"加载摘要: {args.abstracts_file}")
    abstracts = load_abstracts(args.abstracts_file)
    print(f"共 {len(abstracts)} 个摘要")

    if not abstracts:
        print("没有摘要数据")
        return

    # 编码摘要
    embeddings = encode_abstracts(abstracts, args.model)
    print(f"向量维度: {embeddings.shape}")

    # 执行聚类
    labels = cluster_kmeans(embeddings, args.k)

    # 提取关键词
    cluster_keywords = extract_cluster_keywords(abstracts, labels)

    # 提取样本
    result = extract_samples(abstracts, labels, args.samples)

    # 添加关键词到结果
    for cluster_data in result['clusters']:
        cluster_id = cluster_data['cluster_id']
        cluster_data['keywords'] = cluster_keywords.get(cluster_id, [])

    # 保存结果
    print(f"保存结果: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n" + "=" * 70)
    print("完成！")
    print(f"总摘要数: {len(abstracts)}")
    print(f"聚类数: {args.k}")
    print(f"每个簇样本数: {args.samples}")
    label_counts = Counter(labels)
    for cluster_id in sorted(set(labels)):
        count = label_counts[cluster_id]
        keywords = cluster_keywords.get(cluster_id, [])[:3]
        print(f"  簇 {cluster_id:2d}: {count:4d} 篇 | {', '.join(keywords)}")


if __name__ == '__main__':
    main()