"""
使用 spaCy 对 training_data_fixed.json 进行句子级切分
将长文本拆成单句，同时正确映射实体到每个句子中

切分逻辑参考 annotate_papers.py 中的 spaCy 分句模块
"""
import os
import sys
import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict

# ======================== 配置 ========================
INPUT_PATH = os.environ.get("PALEO_INPUT_PATH", "data/annotation_raw.json")
OUTPUT_PATH = os.environ.get("PALEO_OUTPUT_PATH", "data/training_original.json")

# ======================== SpaCy 分句模块（来自 annotate_papers.py） ========================
PALEO_ENTITY_PATTERNS = [
    r'[A-Z][a-z]+\s+[a-z]+\s+sp\.',
    r'[A-Z][a-z]+\s+[a-z]+\s+sp\.\s*n\.',
    r'[A-Z][a-z]+\s+[a-z]+\s+gen\.\s*n\.',
    r'[A-Z]\.\s*[a-z]+',
    r'[A-Z][a-z]+\?\s*[a-z]+',
    r'[A-Z][a-z]+\s+cf\.\s*[a-z]+',
    r'[A-Z][a-z]+\s+aff\.\s*[a-z]+',
    r'[A-Z][a-z]+\s+[a-z]+\s+spp\.',
    r'[A-Z][a-z]+\s+[a-z]+\s+subsp\.',
    r'Fig\.\s*\d+\.?\d*',
    r'pl\.\s*\d+',
    r'No\.\s*\d+',
    r'p\.\s*\d+',
    r'vol\.\s*\d+',
]
ENTITY_PROTECTION_PATTERN = re.compile('|'.join(PALEO_ENTITY_PATTERNS))

GEO_ABBREVIATIONS = [
    'e.g.', 'i.e.', 'cf.', 'sp.', 'gen.', 'etc.', 'aff.', 'subsp.', 'spp.',
    'var.', 'f.', 'No.', 'Fig.', 'pl.', 'p.', 'vol.',
    'al.', 'Dr.', 'Prof.', 'St.', 'Mt.', 'Fm.', 'Grp.', 'Mbr.',
]

_spacy_nlp = None


def _load_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return True
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        print("  spaCy 分句模型加载成功")
        return True
    except (ImportError, OSError) as e:
        print(f"  spaCy 模型加载失败: {e}")
        print("  安装方法: pip install spacy && python -m spacy download en_core_web_sm")
        return False


def _protect_entities(text: str) -> Tuple[str, Dict[str, str]]:
    placeholders = {}
    counter = [0]
    def replace(match):
        placeholder = f"XENTX{counter[0]}XENTX"
        placeholders[placeholder] = match.group()
        counter[0] += 1
        return placeholder
    protected = ENTITY_PROTECTION_PATTERN.sub(replace, text)
    return protected, placeholders


def _restore_entities(text: str, placeholders: Dict[str, str]) -> str:
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)
    return text


def _is_abbreviation_dot(text: str, dot_pos: int) -> bool:
    for abbr in GEO_ABBREVIATIONS:
        dot_offsets = [i for i, c in enumerate(abbr) if c == '.']
        for offset in dot_offsets:
            start = dot_pos - offset
            if start >= 0 and start + len(abbr) <= len(text):
                if text[start:start + len(abbr)] == abbr:
                    return True
    if dot_pos > 0 and dot_pos < len(text) - 1:
        if text[dot_pos - 1].isdigit() and text[dot_pos + 1].isdigit():
            return True
    if dot_pos > 0 and text[dot_pos - 1].isupper():
        if dot_pos < 2 or text[dot_pos - 2] in ' \t\n([':
            rest = text[dot_pos + 1:].lstrip()
            if rest and rest[0].islower():
                return True
    return False


def split_sentences_spacy(text: str) -> List[Dict]:
    """使用 spaCy 分句，返回 [{'text': ..., 'start': ..., 'end': ...}]"""
    protected_text, placeholders = _protect_entities(text)
    doc = _spacy_nlp(protected_text)
    sentences = []
    current_pos = 0
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sent_text = _restore_entities(sent_text, placeholders)
        start = text.find(sent_text, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(sent_text)
        sentences.append({'text': sent_text, 'start': start, 'end': end})
        current_pos = end
    return sentences


def split_sentences_rule(text: str) -> List[Dict]:
    """规则降级分句"""
    sentences = []
    sentence_endings = []
    i = 0
    while i < len(text):
        if text[i] in '.!?':
            if text[i] == '.' and _is_abbreviation_dot(text, i):
                i += 1
                continue
            j = i + 1
            while j < len(text) and text[j] in '.!?)\'"»\u201d\u2019':
                j += 1
            sentence_endings.append(j)
            i = j
        else:
            i += 1
    current_start = 0
    for end_pos in sentence_endings:
        if end_pos > current_start:
            sentence = text[current_start:end_pos]
            if sentence.strip():
                sentences.append({'text': sentence, 'start': current_start, 'end': end_pos})
            current_start = end_pos
    if current_start < len(text):
        remaining = text[current_start:]
        if remaining.strip():
            sentences.append({'text': remaining, 'start': current_start, 'end': len(text)})
    return sentences


def split_sentences(text: str) -> List[Dict]:
    if _spacy_nlp is not None:
        return split_sentences_spacy(text)
    return split_sentences_rule(text)


# ======================== 实体映射到句子 ========================
def map_entities_to_sentence(sent_start: int, sent_end: int, entities: List[Dict]) -> List[Dict]:
    """将原始实体映射到句子中，调整 start/end 为句子内的相对位置
    
    规则：
    - 实体完全在句子内 → 保留，偏移调整为相对位置
    - 实体跨越句子边界 → 保留，截断到句子范围内
    - 实体完全在句子外 → 不保留
    """
    sent_entities = []
    for ent in entities:
        ent_start = ent['start']
        ent_end = ent['end']
        ent_text = ent['text']
        ent_type = ent['entity_type']
        confidence = ent.get('confidence', 95)

        # 实体与句子的重叠区间
        overlap_start = max(ent_start, sent_start)
        overlap_end = min(ent_end, sent_end)

        if overlap_start >= overlap_end:
            # 无重叠
            continue

        # 计算在句子中的相对位置
        new_start = overlap_start - sent_start
        new_end = overlap_end - sent_start

        # 获取实际文本
        new_text = ent_text[overlap_start - ent_start : overlap_end - ent_start] if (overlap_start > ent_start or overlap_end < ent_end) else ent_text

        # 验证文本匹配
        sent_text_region = None  # 延迟获取
        
        sent_entities.append({
            'start': new_start,
            'end': new_end,
            'entity_type': ent_type,
            'text': new_text,
            'confidence': confidence,
        })

    return sent_entities


# ======================== 主流程 ========================
def segment_data(data: List[Dict]) -> List[Dict]:
    """对每条数据进行句子级切分，保留实体映射"""
    segmented = []
    stats = {
        'original_count': len(data),
        'original_entities': 0,
        'segmented_count': 0,
        'segmented_entities': 0,
        'entities_dropped': 0,
        'no_entity_sentences': 0,
    }

    for idx, item in enumerate(data):
        text = item.get('text', '')
        entities = item.get('entities', [])
        source = item.get('source', '')
        data_type = item.get('data_type', '')
        cluster = item.get('cluster', 0)

        stats['original_entities'] += len(entities)

        # 分句
        sentences = split_sentences(text)

        if not sentences:
            # 分句失败，保留原文
            segmented.append({
                'text': text,
                'entities': entities,
                'source': source,
                'data_type': data_type,
                'cluster': cluster,
            })
            stats['segmented_count'] += 1
            stats['segmented_entities'] += len(entities)
            continue

        # 每个句子生成一条记录
        for sent in sentences:
            sent_text = sent['text'].strip()
            if not sent_text:
                continue

            # 过滤太短的句子（如纯数字、纯引用标记等）
            if len(sent_text) < 10:
                continue

            # 映射实体
            sent_entities = map_entities_to_sentence(sent['start'], sent['end'], entities)

            # 验证实体文本
            valid_entities = []
            for ent in sent_entities:
                if ent['start'] < 0 or ent['end'] > len(sent_text) or ent['start'] >= ent['end']:
                    stats['entities_dropped'] += 1
                    continue
                actual = sent_text[ent['start']:ent['end']]
                if actual != ent['text']:
                    # 文本不匹配，尝试修正
                    pos = sent_text.find(ent['text'])
                    if pos != -1:
                        ent['start'] = pos
                        ent['end'] = pos + len(ent['text'])
                        valid_entities.append(ent)
                    else:
                        stats['entities_dropped'] += 1
                else:
                    valid_entities.append(ent)

            stats['segmented_count'] += 1
            stats['segmented_entities'] += len(valid_entities)
            if not valid_entities:
                stats['no_entity_sentences'] += 1

            segmented.append({
                'text': sent_text,
                'entities': valid_entities,
                'source': source,
                'data_type': data_type,
                'cluster': cluster,
            })

    return segmented, stats


def main():
    print("=" * 60)
    print("句子级切分 training_data_fixed.json")
    print("=" * 60)

    # 加载 spaCy
    spacy_ok = _load_spacy()
    method = "spaCy" if spacy_ok else "规则降级"
    print(f"  分句方法: {method}")

    # 加载数据
    print(f"\n正在读取: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  原始数据: {len(data)} 条")

    # 切分
    print("\n开始切分...")
    segmented, stats = segment_data(data)

    # 统计
    print(f"\n{'=' * 60}")
    print("切分结果统计")
    print(f"{'=' * 60}")
    print(f"原始记录数:         {stats['original_count']}")
    print(f"原始实体数:         {stats['original_entities']}")
    print(f"切分后记录数:       {stats['segmented_count']}")
    print(f"切分后实体数:       {stats['segmented_entities']}")
    print(f"实体丢失数:         {stats['entities_dropped']}")
    print(f"无实体句子数:       {stats['no_entity_sentences']}")
    print(f"有实体句子数:       {stats['segmented_count'] - stats['no_entity_sentences']}")
    print(f"实体保留率:         {stats['segmented_entities']/max(stats['original_entities'],1)*100:.1f}%")

    # 文本长度分布
    text_lens = [len(item['text']) for item in segmented]
    ent_counts = [len(item['entities']) for item in segmented]
    print(f"\n文本长度: min={min(text_lens)}, max={max(text_lens)}, avg={sum(text_lens)/len(text_lens):.0f}")
    print(f"实体数:   min={min(ent_counts)}, max={max(ent_counts)}, avg={sum(ent_counts)/len(ent_counts):.1f}")

    # 实体类型分布
    type_counts = defaultdict(int)
    for item in segmented:
        for ent in item['entities']:
            type_counts[ent['entity_type']] += 1
    total_ents = sum(type_counts.values())
    print(f"\n实体类型分布:")
    for etype in ['taxa', 'location', 'age', 'strata', 'lithology', 'facies', 'section']:
        count = type_counts.get(etype, 0)
        pct = count / total_ents * 100 if total_ents > 0 else 0
        print(f"  {etype}: {count} ({pct:.1f}%)")

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(segmented, f, ensure_ascii=False, indent=2)
    print(f"\n切分后数据已保存到: {OUTPUT_PATH}")

    # 验证
    print("\n验证切分后数据的实体一致性...")
    errors = 0
    for item in segmented:
        text = item['text']
        for ent in item['entities']:
            if 0 <= ent['start'] < ent['end'] <= len(text):
                if text[ent['start']:ent['end']] != ent['text']:
                    errors += 1
            else:
                errors += 1
    print(f"  验证结果: {'通过' if errors == 0 else f'{errors} 个错误'}")


if __name__ == "__main__":
    main()
