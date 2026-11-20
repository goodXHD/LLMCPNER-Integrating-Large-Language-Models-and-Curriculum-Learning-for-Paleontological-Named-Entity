"""
NER v2 数据增广脚本 (外部 API 版)
====================================
基于 augment_v2.py 的增广思路，使用 api.n1n.ai 外部 LLM

三种策略：
1. 实体替换 - 从原始样本替换部分实体
2. 原文改写 - 改写原始文本，替换所有实体为不同实体，上百种改写指令保证风格多样性
3. 实体组合+风格生成 - 按排序组合实体 + 随机选择生成风格，从零生成

安全特性：
- 测试模式：先验证 API 连通性和生成效果
- 断点续传：每条成功立即保存，支持恢复
- 余额不足检测：402/429/quota 错误自动停止+保存
- 信号处理：Ctrl+C 优雅保存

"""

import json
import random
import time
import re
import argparse
import os
import signal
import threading
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ==================== 配置 ====================
FIXED_DATA_PATH = os.environ.get("PALEO_FIXED_DATA_PATH", "data/training_original.json")
CLEANED_MERGED_PATH = os.environ.get("PALEO_CLEANED_MERGED_PATH", "data/cleaned_merged_entities.xlsx")
CLEANED_ENTITIES_ALL_PATH = os.environ.get("PALEO_CLEANED_ENTITIES_ALL_PATH", "data/cleaned_entities_all.xlsx")
OUTPUT_DIR = os.environ.get("PALEO_OUTPUT_DIR", "outputs/augmentation")

DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")
MAX_WORKERS = 50
RETRY_TIMES = 3

TARGET_TEXT_LENGTH_MIN = 5
TARGET_TEXT_LENGTH_MAX = 2000


def compute_target_length(num_entities):
    if num_entities <= 1:
        return (30, 250)
    elif num_entities <= 3:
        return (60, 350)
    elif num_entities <= 6:
        return (100, 500)
    elif num_entities <= 9:
        return (200, 650)
    elif num_entities <= 12:
        return (300, 800)
    else:
        return (400, 1200)


# 实体类型描述
ENTITY_TYPE_DESCRIPTIONS = {
    "taxa": "Species/taxa: Biological species names in paleontology, including genus, species, abbreviated forms (e.g., T. rex, A. sp. nov., B. cf. C), indeterminate species (e.g., Trilobita indet.), and new species designations",
    "location": "Location: Geographic place names mentioned in the text, including countries, provinces, cities, regions, mountain ranges, basins, and directional modifiers (e.g., South China, western Sichuan Basin)",
    "section": "Section: Named geological sections or outcrop localities where fossils or rock samples were collected (e.g., Wa'ergang Section, Xiaoheba Section)",
    "strata": "Strata: Lithostratigraphic unit names, including Formation, Group, Member, Bed (e.g., Guole Formation, Niutitang Group)",
    "lithology": "Lithology: Rock type descriptions, including sedimentary, igneous, and metamorphic rock names and their characteristics (e.g., limestone, shale, sandstone, mudstone, dolomite)",
    "facies": "Facies: Depositional or sedimentary facies descriptions representing specific sedimentary environments (e.g., shallow marine facies, deltaic facies, deep-water facies)",
    "age": "Age: Geological time terms, including periods, epochs, ages, and time ranges (e.g., Cambrian, Early Ordovician, Late Permian, Middle Triassic)"
}

ENTITY_TYPE_WEIGHTS = {
    "taxa": 43, "location": 24, "age": 18,
    "strata": 5, "lithology": 5, "facies": 3, "section": 2
}

# ============================================================
# 改写指令 (100+)
# ============================================================
PARAPHRASE_INSTRUCTIONS = [
    # ---- 以不同实体类型为主语 (20+) ----
    "Rewrite with a species name as the grammatical subject of the main clause",
    "Rewrite with a geographic location as the grammatical subject",
    "Rewrite with a stratigraphic unit as the grammatical subject",
    "Rewrite with a geological age as the grammatical subject",
    "Rewrite with a rock type (lithology) as the grammatical subject",
    "Rewrite with a sedimentary facies as the grammatical subject",
    "Rewrite with a geological section as the grammatical subject",
    "Rewrite making the fossil discovery the main topic",
    "Rewrite making the sedimentary environment the main topic",
    "Rewrite making the stratigraphic correlation the main topic",
    "Rewrite making the paleogeographic setting the main topic",
    "Rewrite making the depositional conditions the main topic",
    "Rewrite focusing on the taxonomic identification",
    "Rewrite focusing on the biostratigraphic significance",
    "Rewrite focusing on the lithostratigraphic framework",
    "Rewrite from the perspective of the rock unit being described",
    "Rewrite from the perspective of the fossil organism's habitat",
    "Rewrite as if describing the geological history of the region",
    "Rewrite as if narrating the process of fossil discovery",
    "Rewrite emphasizing the temporal sequence of events",

    # ---- 不同的语态和句式结构 (20+) ----
    "Rewrite entirely in passive voice",
    "Rewrite entirely in active voice",
    "Rewrite using a conditional structure (if...then...)",
    "Rewrite using a contrastive structure (whereas, while, although)",
    "Rewrite using a causative structure (because, due to, resulting from)",
    "Rewrite as a compound sentence with two independent clauses joined by a semicolon",
    "Rewrite as a complex sentence with a subordinate clause first",
    "Rewrite using an inverted sentence structure",
    "Rewrite using a participial phrase to open the sentence",
    "Rewrite starting with a prepositional phrase indicating location",
    "Rewrite starting with a prepositional phrase indicating time",
    "Rewrite using a relative clause to add description",
    "Rewrite as a rhetorical question followed by an answer",
    "Rewrite using parallel structure for multiple entities",
    "Rewrite as a definition-style statement (X is defined as...)",
    "Rewrite using appositive phrases to rename entities",
    "Rewrite with the main verb at the beginning (VOS word order variation)",
    "Rewrite using a cleft sentence structure (It is X that...)",
    "Rewrite using a dangling modifier style opening",
    "Rewrite with multiple short clauses connected by conjunctions",

    # ---- 不同的文体风格 (20+) ----
    "Rewrite in a concise field note style, telegraphic and brief",
    "Rewrite in an elaborate descriptive style with rich adjectives",
    "Rewrite in a formal taxonomic description style",
    "Rewrite in a comparative analysis style",
    "Rewrite in a synthesis or review style",
    "Rewrite in a hypothesis-testing narrative style",
    "Rewrite in a chronological narrative style",
    "Rewrite in a technical sedimentological report style",
    "Rewrite in a museum catalog entry style",
    "Rewrite in a geological survey report style",
    "Rewrite in a paleoecological interpretation style",
    "Rewrite in a stratigraphic correlation discussion style",
    "Rewrite in a biostratigraphic zonation style",
    "Rewrite in an expedition log style",
    "Rewrite in a scientific abstract style",
    "Rewrite in a dissertation results section style",
    "Rewrite in a journal article introduction style",
    "Rewrite in a journal article discussion style",
    "Rewrite in a conference presentation transcript style",
    "Rewrite in a textbook explanation style",

    # ---- 不同的信息排列方式 (20+) ----
    "Rewrite presenting information from general to specific",
    "Rewrite presenting information from specific to general",
    "Rewrite presenting the oldest geological information first, then progressively younger",
    "Rewrite presenting the youngest geological information first, then progressively older",
    "Rewrite starting with the biological findings, then the geological context",
    "Rewrite starting with the geological context, then the biological findings",
    "Rewrite starting with the geographic setting, then the stratigraphic framework",
    "Rewrite starting with the stratigraphic framework, then the geographic setting",
    "Rewrite starting with conclusions, then supporting evidence",
    "Rewrite starting with observations, then interpretations",
    "Rewrite listing entities in order of abundance or significance",
    "Rewrite organizing by spatial relationships (from bottom to top of the stratigraphic column)",
    "Rewrite organizing by spatial relationships (from top to bottom of the stratigraphic column)",
    "Rewrite organizing from the largest geographic scale to the smallest",
    "Rewrite organizing from the smallest geographic scale to the largest",
    "Rewrite presenting the depositional environment first, then the fossils within it",
    "Rewrite presenting the fossils first, then inferring the depositional environment",
    "Rewrite following a logical argument structure: observation → inference → conclusion",
    "Rewrite with background information first, then the new finding",
    "Rewrite with the key new finding first, then background context",

    # ---- 不同的学术表达方式 (20+) ----
    "Rewrite using hedging language (may, might, could, appears to, seems to)",
    "Rewrite using assertive, confident language (demonstrates, confirms, establishes)",
    "Rewrite using speculative language (suggests, implies, raises the possibility)",
    "Rewrite using quantifying language where possible",
    "Rewrite emphasizing certainty and evidence-based reasoning",
    "Rewrite using transitional phrases for flow (furthermore, moreover, additionally)",
    "Rewrite using concessive phrases (nevertheless, notwithstanding, despite)",
    "Rewrite using exemplification (for instance, such as, notably)",
    "Rewrite using summative language (in summary, overall, collectively)",
    "Rewrite with minimal use of the verb 'to be' (is, are, was, were)",
    "Rewrite using predominantly action verbs (yielded, deposited, preserved, inhabited)",
    "Rewrite using nominalization (convert verbs to noun phrases)",
    "Rewrite using more adverbial modifiers for precision",
    "Rewrite using parenthetical asides for supplementary information",
    "Rewrite with parenthetical citations style embedded in the text",
    "Rewrite using dash-separated parenthetical remarks",
    "Rewrite using a colon to introduce a list or explanation",
    "Rewrite incorporating cross-reference language (see above, as noted, refer to)",
    "Rewrite using footnoting-style phrases (it should be noted that, importantly)",
    "Rewrite using measured, conservative academic language",

    # ---- 不同的修辞手法 (20+) ----
    "Rewrite using an analogy to explain the geological process",
    "Rewrite using metaphorical language for the depositional environment",
    "Rewrite using understatement for dramatic effect",
    "Rewrite with a tone of scientific excitement about the discovery",
    "Rewrite with a tone of scientific caution and restraint",
    "Rewrite as if writing for a popular science audience",
    "Rewrite as if writing for specialist colleagues only",
    "Rewrite as if the findings challenge a previous paradigm",
    "Rewrite as if the findings confirm a long-held hypothesis",
    "Rewrite as if describing a rare or unusual occurrence",
    "Rewrite as if describing a typical or representative example",
    "Rewrite using enumeration (firstly, secondly, finally)",
    "Rewrite using a problem-solution structure",
    "Rewrite using a claim-evidence-reasoning framework",
    "Rewrite using vivid sensory descriptions of the field site",
    "Rewrite using precise technical terminology throughout",
    "Rewrite using simpler, more accessible language",
    "Rewrite with an emphasis on the novelty of the finding",
    "Rewrite with an emphasis on the broader implications",
    "Rewrite with an emphasis on methodology and analytical approach",
]

# ============================================================
# 实体排列规则 - 策略3专用
# ============================================================
ENTITY_ORDER_RULES = [
    {"name": "taxa-centric", "rule": "taxa first, then its geological context, then geographic context",
     "priority": ["taxa", "age", "strata", "lithology", "facies", "location", "section"]},
    {"name": "location-centric", "rule": "start from the geographic setting, narrow down to details",
     "priority": ["location", "section", "strata", "lithology", "facies", "age", "taxa"]},
    {"name": "strata-centric", "rule": "stratigraphy as the main framework, then fill in details",
     "priority": ["strata", "age", "lithology", "facies", "taxa", "location", "section"]},
    {"name": "age-centric", "rule": "establish the time frame first, then describe what exists in that time",
     "priority": ["age", "strata", "lithology", "facies", "location", "taxa", "section"]},
    {"name": "lithology-centric", "rule": "describe the rocks first, then their context",
     "priority": ["lithology", "facies", "strata", "age", "taxa", "location", "section"]},
    {"name": "facies-centric", "rule": "depositional environment as the main theme",
     "priority": ["facies", "lithology", "strata", "age", "taxa", "location", "section"]},
    {"name": "discovery-narrative", "rule": "narrate a discovery: where → what rocks → what fossils → when",
     "priority": ["location", "section", "lithology", "facies", "strata", "taxa", "age"]},
    {"name": "causal-chain", "rule": "causal chain: time → environment → rocks → fossils",
     "priority": ["age", "facies", "lithology", "strata", "taxa", "location", "section"]},
    {"name": "spatial-large-to-small", "rule": "spatial scale from large to small: region → basin → section → bed → fossil",
     "priority": ["location", "section", "strata", "lithology", "facies", "taxa", "age"]},
    {"name": "spatial-small-to-large", "rule": "spatial scale from small to large: fossil → bed → section → region",
     "priority": ["taxa", "lithology", "facies", "strata", "section", "location", "age"]},
    {"name": "temporal-narrow-to-broad", "rule": "time from specific to general: exact age → period → era",
     "priority": ["taxa", "strata", "lithology", "age", "facies", "location", "section"]},
    {"name": "ecological-context", "rule": "organism → its environment → its geological record → geographic location",
     "priority": ["taxa", "facies", "lithology", "strata", "age", "location", "section"]},
    {"name": "interleave-context-entity", "rule": "alternate between context types and entity types",
     "priority": ["age", "taxa", "location", "taxa", "strata", "taxa", "lithology"]},
    {"name": "group-by-domain", "rule": "group geological terms together, then biological terms",
     "priority": ["strata", "age", "lithology", "facies", "taxa", "location", "section"]},
    {"name": "contrast-pairs", "rule": "present contrasting information: location vs taxa, age vs lithology",
     "priority": ["location", "taxa", "age", "lithology", "strata", "facies", "section"]},
    {"name": "natural-flow", "rule": "as if naturally flowing in a paragraph without deliberate ordering",
     "priority": None},
    {"name": "random", "rule": "entities appear in no particular order",
     "priority": None},
    {"name": "taxa-diversity", "rule": "multiple taxa listed together to show biodiversity, then shared context",
     "priority": ["taxa", "taxa", "age", "strata", "lithology", "facies", "location"]},
    {"name": "multi-location", "rule": "compare across multiple locations, listing each location with its associated entities",
     "priority": ["location", "strata", "taxa", "location", "strata", "taxa", "age"]},
    {"name": "stratigraphic-column", "rule": "describe a stratigraphic column top to bottom with multiple strata and lithologies",
     "priority": ["age", "strata", "lithology", "strata", "lithology", "taxa", "location"]},
    {"name": "age-range", "rule": "describe a time range spanning multiple ages with corresponding taxa and strata",
     "priority": ["age", "age", "taxa", "strata", "taxa", "lithology", "location"]},
    {"name": "lithological-variation", "rule": "describe lithological variation within a unit with multiple rock types",
     "priority": ["strata", "lithology", "lithology", "facies", "taxa", "age", "location"]},
]

# ============================================================
# 生成风格指令 - 策略3专用
# ============================================================
GENERATION_STYLES = [
    # ---- 学术文体 (30+) ----
    "Write as a formal taxonomic description",
    "Write as a stratigraphic correlation statement",
    "Write as a paleoecological interpretation",
    "Write as a sedimentological observation",
    "Write as a biostratigraphic zonation note",
    "Write as a geological field report entry",
    "Write as a museum specimen catalog description",
    "Write as a journal abstract summary",
    "Write as a dissertation results paragraph",
    "Write as a conference abstract",
    "Write as a research letter",
    "Write as a technical stratigraphic note",
    "Write as a paleobiogeographic analysis",
    "Write as a basin analysis summary",
    "Write as a sequence stratigraphy interpretation",
    "Write as a chemostratigraphic correlation note",
    "Write as a magnetostratigraphic summary",
    "Write as a lithostratigraphic boundary description",
    "Write as a facies model description",
    "Write as a depositional system interpretation",
    "Write as a regional geology synthesis",
    "Write as a paleoenvironmental reconstruction",
    "Write as a taphonomic observation",
    "Write as a diagenetic description",
    "Write as a petrographic description",
    "Write as a well log interpretation note",
    "Write as a seismic stratigraphy note",
    "Write as a structural geology observation",
    "Write as a geomorphological description",
    "Write as a hydrogeological note",
    # ---- 句式结构 (30+) ----
    "Use a single long sentence with multiple clauses",
    "Use two medium sentences",
    "Use three short punchy sentences",
    "Use a topic sentence followed by supporting details",
    "Open with a prepositional phrase of location",
    "Open with a prepositional phrase of time",
    "Open with a participial phrase",
    "Open with an adverbial clause",
    "Use an inverted sentence structure",
    "Use a cleft sentence (It was in X that Y was found)",
    "Use a passive construction throughout",
    "Use an active construction throughout",
    "Use a parallel structure listing multiple findings",
    "Use a cause-and-effect chain",
    "Use a contrastive structure (while X, Y)",
    "Use a comparative structure (unlike X, Y)",
    "Use a conditional structure (if X, then Y)",
    "Use a concessive structure (although X, Y)",
    "Frame as an observation then interpretation",
    "Frame as evidence leading to a conclusion",
    "Use a definitional structure (X, a Y, is...)",
    "Use an enumerative structure (first, second, third)",
    "Use an illustrative structure (for example, notably)",
    "Use a summarizing structure (overall, collectively)",
    "Use a hypothetical structure (it is possible that)",
    "Use a deductive structure (given X, therefore Y)",
    "Use an inductive structure (X, Y, Z suggest that...)",
    "Use a chronological narrative",
    "Use a spatial description (from base to top)",
    "Use a functional description (the role of X in Y)",
    # ---- 语气与强调 (20+) ----
    "Emphasize the uniqueness of the finding",
    "Emphasize the typicality of the occurrence",
    "Emphasize the temporal precision of the dating",
    "Emphasize the geographic breadth of the distribution",
    "Emphasize the taxonomic diversity",
    "Emphasize the sedimentological detail",
    "Emphasize the environmental interpretation",
    "Use confident, assertive language",
    "Use cautious, hedging language",
    "Use enthusiastic language about the discovery",
    "Use matter-of-fact, neutral language",
    "Write as if challenging an existing theory",
    "Write as if confirming a previous hypothesis",
    "Write as if providing the first documentation",
    "Write as if expanding on previous work",
    "Use precise quantitative language where appropriate",
    "Use qualitative descriptive language",
    "Use technical jargon appropriate for specialists",
    "Use accessible language for a broader audience",
    "Write with a narrative storytelling quality",
    # ---- 内容角度 (20+) ----
    "Focus on what was found (the fossils/rocks)",
    "Focus on where it was found (the geography)",
    "Focus on when it was formed (the geological age)",
    "Focus on how it was formed (the depositional process)",
    "Focus on why it matters (the scientific significance)",
    "Describe the fossil assemblage and its implications",
    "Describe the rock sequence and its interpretation",
    "Describe the geographic setting and its influence",
    "Describe the temporal framework and its constraints",
    "Describe the depositional environment and its characteristics",
    "Frame as a biostratigraphic event",
    "Frame as a lithostratigraphic boundary",
    "Frame as a paleobiogeographic observation",
    "Frame as a sequence stratigraphic surface",
    "Frame as a taphonomic window",
    "Frame as a paleoclimatic indicator",
    "Frame as a paleoceanographic signal",
    "Frame as a basin evolution milestone",
    "Frame as an evolutionary radiation event",
    "Frame as a mass extinction survivor assemblage",
]


# ============================================================
# 全局停止控制
# ============================================================
_stop_event = threading.Event()

def should_stop():
    return _stop_event.is_set()

def request_stop(reason):
    print(f"\n!!! 停止请求: {reason} !!!", flush=True)
    _stop_event.set()


# ============================================================
# LLM 调用 (OpenAI SDK) — 全局客户端复用
# ============================================================
_global_client = None
_client_lock = threading.Lock()

def _get_client(api_base=DEFAULT_API_BASE, api_key=DEFAULT_API_KEY):
    """获取或创建全局 OpenAI 客户端（线程安全、连接复用）。"""
    global _global_client
    if _global_client is None:
        with _client_lock:
            if _global_client is None:
                from openai import OpenAI
                _global_client = OpenAI(base_url=api_base, api_key=api_key)
    return _global_client


def call_llm(prompt, max_retries=RETRY_TIMES, api_base=DEFAULT_API_BASE,
             api_key=DEFAULT_API_KEY, model=DEFAULT_MODEL, temperature=0.8, max_tokens=2048):
    """调用 OpenAI 兼容 API，带重试和余额不足检测。复用全局客户端。"""
    if should_stop():
        return None

    client = _get_client(api_base, api_key)

    for attempt in range(max_retries):
        if should_stop():
            return None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            is_quota = any(kw in err for kw in [
                'quota', 'insufficient_quota', 'billing', '429', '402',
                'rate_limit', 'capacity', 'credit', 'balance',
            ])
            if is_quota:
                request_stop(f"API 余额不足或限额 (错误: {e})")
                return None
            # 连接错误时重置客户端，强制重建连接
            is_conn = any(kw in err for kw in ['connection', 'connect', 'refused', 'timeout', 'timed out'])
            if is_conn:
                global _global_client
                with _client_lock:
                    _global_client = None
                client = _get_client(api_base, api_key)
            if attempt < max_retries - 1:
                wait = [2, 5, 10][min(attempt, 2)]
                print(f"    API 调用失败，{wait}s 后重试 ({attempt+1}/{max_retries}): {e}", flush=True)
                time.sleep(wait)

    return None


# ============================================================
# 数据加载
# ============================================================
def load_original_data():
    with open(FIXED_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    valid = [item for item in data if item.get('entities') and len(item.get('text', '')) >= 30]
    print(f"  加载修复数据: {len(data)} 条, 有效: {len(valid)} 条")
    return valid


def load_entities_from_excel():
    import pandas as pd

    entities = {
        "taxa": [], "location": [], "section": [],
        "strata": [], "lithology": [], "facies": [], "age": []
    }

    LOCATION_MODIFIERS = [
        "", "the ", "Eastern ", "Western ", "Northern ", "Southern ", "Central ",
        "Northeastern ", "Northwestern ", "Southeastern ", "Southwestern ",
        "eastern ", "western ", "northern ", "southern ", "central ",
        "eastern part of ", "western part of ", "northern part of ", "southern part of "
    ]
    SECTION_SUFFIXES = ["Section", "section"]
    STRATA_SUFFIXES = ["Formation", "formation", "Group", "group", "Bed", "bed", "Member", "member"]

    if os.path.exists(CLEANED_MERGED_PATH):
        print(f"  加载 {CLEANED_MERGED_PATH} ...")
        df1 = pd.read_excel(CLEANED_MERGED_PATH)
        for _, row in df1.iterrows():
            entity_type = row.get('entity_type', '')
            if entity_type == 'taxa':
                if pd.notna(row.get('cleaned')) and row.get('cleaned'):
                    entities['taxa'].append(row['cleaned'])
                if pd.notna(row.get('abbreviated')) and row.get('abbreviated'):
                    abbrev = row['abbreviated']
                    if '.' in str(abbrev) and not str(abbrev).endswith('. sp'):
                        entities['taxa'].append(abbrev)
                for field in ['sp_nov', 'cf_sp', 'indet', 'genus_sp']:
                    val = row.get(field, '')
                    if pd.notna(val) and val:
                        entities['taxa'].append(val)
            elif entity_type == 'location':
                if pd.notna(row.get('cleaned')) and row.get('cleaned'):
                    entities['location'].append(row['cleaned'])
                if pd.notna(row.get('country')) and row.get('country'):
                    for mod in LOCATION_MODIFIERS:
                        entities['location'].append(mod + row['country'])
                if pd.notna(row.get('province')) and row.get('province'):
                    entities['location'].append(row['province'])
                    for mod in LOCATION_MODIFIERS[:8]:
                        entities['location'].append(mod + row['province'])
                if pd.notna(row.get('city')) and row.get('city'):
                    entities['location'].append(row['city'])
                if pd.notna(row.get('chain')) and row.get('chain'):
                    chain = row['chain']
                    if '|' in str(chain):
                        for part in chain.split('|'):
                            entities['location'].append(part.strip())
                    else:
                        entities['location'].append(chain)
            elif entity_type == 'section':
                sn = row.get('section_name', '')
                if pd.notna(sn) and sn:
                    for suffix in SECTION_SUFFIXES:
                        if str(sn).endswith('Section') or str(sn).endswith('section'):
                            base = str(sn).rsplit(' ', 1)[0]
                            entities['section'].append(f"{base} {suffix}")
                        else:
                            entities['section'].append(f"{sn} {suffix}")
            elif entity_type == 'strata':
                fn = row.get('full_name', '')
                if pd.notna(fn) and fn:
                    entities['strata'].append(fn)
                    for ns in STRATA_SUFFIXES:
                        for old_sfx in ["Formation", "formation", "Group", "group", "Member", "member", "Bed", "bed"]:
                            if fn.endswith(' ' + old_sfx):
                                base = fn[:-len(old_sfx)].strip()
                                entities['strata'].append(f"{base} {ns}")
                                break
            elif entity_type == 'lithology':
                if pd.notna(row.get('cleaned')) and row.get('cleaned'):
                    entities['lithology'].append(row['cleaned'])
                for field in ['synonyms', 'related']:
                    val = row.get(field, '')
                    if pd.notna(val) and val:
                        for item in str(val).split('|'):
                            item = item.strip()
                            if item:
                                entities['lithology'].append(item)
            elif entity_type == 'facies':
                if pd.notna(row.get('cleaned')) and row.get('cleaned'):
                    entities['facies'].append(row['cleaned'])
                val = row.get('synonyms', '')
                if pd.notna(val) and val:
                    for item in str(val).split('|'):
                        item = item.strip()
                        if item:
                            entities['facies'].append(item)
            elif entity_type == 'age':
                if pd.notna(row.get('cleaned')) and row.get('cleaned'):
                    entities['age'].append(row['cleaned'])
                for field in ['period', 'epoch', 'age_range']:
                    val = row.get(field, '')
                    if pd.notna(val) and val:
                        entities['age'].append(val)

    if os.path.exists(CLEANED_ENTITIES_ALL_PATH):
        print(f"  加载 {CLEANED_ENTITIES_ALL_PATH} ...")
        df2 = pd.read_excel(CLEANED_ENTITIES_ALL_PATH)
        for _, row in df2.iterrows():
            if pd.notna(row.get('taxa_cleaned')) and row.get('taxa_cleaned'):
                entities['taxa'].append(row['taxa_cleaned'])
            if pd.notna(row.get('taxa_abbreviated')) and row.get('taxa_abbreviated'):
                abbrev = row['taxa_abbreviated']
                if '.' in str(abbrev) and not str(abbrev).endswith('. sp'):
                    entities['taxa'].append(abbrev)
            for field in ['taxa_sp_nov', 'taxa_cf_sp', 'taxa_indet', 'taxa_genus_sp']:
                val = row.get(field, '')
                if pd.notna(val) and val:
                    entities['taxa'].append(val)
            if pd.notna(row.get('location_cleaned')) and row.get('location_cleaned'):
                loc = row['location_cleaned']
                entities['location'].append(loc)
                for mod in LOCATION_MODIFIERS[:8]:
                    entities['location'].append(mod + loc)
            if pd.notna(row.get('location_country')) and row.get('location_country'):
                for mod in LOCATION_MODIFIERS:
                    entities['location'].append(mod + row['location_country'])
            if pd.notna(row.get('location_province')) and row.get('location_province'):
                entities['location'].append(row['location_province'])
                for mod in LOCATION_MODIFIERS[:8]:
                    entities['location'].append(mod + row['location_province'])
            if pd.notna(row.get('location_city')) and row.get('location_city'):
                entities['location'].append(row['location_city'])
            if pd.notna(row.get('location_chain')) and row.get('location_chain'):
                chain = row['location_chain']
                if '|' in str(chain):
                    for part in chain.split('|'):
                        entities['location'].append(part.strip())
                    entities['location'].append(chain.replace('|', ', '))
                else:
                    entities['location'].append(chain)
            sn = row.get('section_name', '')
            if pd.notna(sn) and sn:
                for suffix in SECTION_SUFFIXES:
                    if str(sn).endswith('Section') or str(sn).endswith('section'):
                        base = str(sn).rsplit(' ', 1)[0]
                        entities['section'].append(f"{base} {suffix}")
                    else:
                        entities['section'].append(f"{sn} {suffix}")
            fn = row.get('strata_full', '')
            if pd.notna(fn) and fn:
                entities['strata'].append(fn)
                for ns in STRATA_SUFFIXES:
                    for old_sfx in ["Formation", "formation", "Group", "group", "Member", "member", "Bed", "bed"]:
                        if fn.endswith(' ' + old_sfx):
                            base = fn[:-len(old_sfx)].strip()
                            entities['strata'].append(f"{base} {ns}")
                            break
            if pd.notna(row.get('lithology_cleaned')) and row.get('lithology_cleaned'):
                entities['lithology'].append(row['lithology_cleaned'])
            for field_key in ['lithology_synonyms', 'lithology_related']:
                val = row.get(field_key, '')
                if pd.notna(val) and val:
                    for item in str(val).split('|'):
                        item = item.strip()
                        if item:
                            entities['lithology'].append(item)
            if pd.notna(row.get('facies_cleaned')) and row.get('facies_cleaned'):
                entities['facies'].append(row['facies_cleaned'])
            val = row.get('facies_synonyms', '')
            if pd.notna(val) and val:
                for item in str(val).split('|'):
                    item = item.strip()
                    if item:
                        entities['facies'].append(item)
            if pd.notna(row.get('age_cleaned')) and row.get('age_cleaned'):
                entities['age'].append(row['age_cleaned'])
            for field_key in ['age_period', 'age_epoch', 'age_range']:
                val = row.get(field_key, '')
                if pd.notna(val) and val:
                    entities['age'].append(val)

    for key in entities:
        entities[key] = list(set([str(e) for e in entities[key] if e and str(e) != 'nan']))

    return entities


def load_entities_from_training_data(original_data):
    entities = defaultdict(set)
    for item in original_data:
        for ent in item.get('entities', []):
            etype = ent.get('entity_type', '')
            etext = ent.get('text', '').strip()
            if etype and etext:
                entities[etype].add(etext)
    return {k: list(v) for k, v in entities.items()}


def merge_entity_pools(pool1, pool2):
    merged = {}
    for key in set(list(pool1.keys()) + list(pool2.keys())):
        s = set()
        if key in pool1:
            s.update(pool1[key])
        if key in pool2:
            s.update(pool2[key])
        merged[key] = list(s)
    return merged


# ============================================================
# 解析 LLM 输出
# ============================================================
def parse_entities_from_text(text):
    pattern = r'\[(\w+)\]([^\[]+)\[/\1\]'
    entities = []
    used_positions = set()

    matches = list(re.finditer(pattern, text))
    clean_text = text
    for match in matches:
        clean_text = clean_text.replace(match.group(0), match.group(2), 1)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    for match in matches:
        entity_type = match.group(1)
        entity_text = match.group(2).strip()

        start = clean_text.find(entity_text)
        while start != -1 and start in used_positions:
            start = clean_text.find(entity_text, start + 1)

        if start != -1:
            used_positions.add(start)
            entities.append({
                "start": start,
                "end": start + len(entity_text),
                "entity_type": entity_type,
                "text": entity_text
            })

    return clean_text, entities


# ============================================================
# 实体采样和排列
# ============================================================
def select_entity_types_weighted():
    all_types = list(ENTITY_TYPE_WEIGHTS.keys())
    all_weights = [ENTITY_TYPE_WEIGHTS[t] for t in all_types]
    num_types = random.choices([1, 2, 3, 4, 5, 6], weights=[10, 15, 25, 25, 15, 10], k=1)[0]
    selected = []
    remaining_types = all_types[:]
    remaining_weights = all_weights[:]
    for _ in range(num_types):
        if not remaining_types:
            break
        chosen = random.choices(remaining_types, weights=remaining_weights, k=1)[0]
        selected.append(chosen)
        idx = remaining_types.index(chosen)
        remaining_types.pop(idx)
        remaining_weights.pop(idx)
    return sorted(selected)


def sample_entities_for_types(entities, entity_types, max_per_type=5):
    result = {}
    for etype in entity_types:
        if etype in entities and entities[etype]:
            available = entities[etype]
            num = random.randint(1, min(max_per_type, len(available)))
            result[etype] = random.sample(available, num)
    return result


def order_entities_by_rule(sampled_entities, rule):
    priority = rule.get("priority")
    if priority is None:
        flat_entities = []
        for etype, texts in sampled_entities.items():
            for text in texts:
                flat_entities.append({"type": etype, "text": text})
        random.shuffle(flat_entities)
    else:
        sorted_entities = []
        used_indices = {}
        for p_type in priority:
            if p_type in sampled_entities:
                idx = used_indices.get(p_type, 0)
                if idx < len(sampled_entities[p_type]):
                    sorted_entities.append({
                        "type": p_type,
                        "text": sampled_entities[p_type][idx]
                    })
                    used_indices[p_type] = idx + 1
        remaining = []
        for etype, texts in sampled_entities.items():
            start_idx = used_indices.get(etype, 0)
            for i in range(start_idx, len(texts)):
                remaining.append({"type": etype, "text": texts[i]})
        if remaining:
            random.shuffle(remaining)
            result = sorted_entities[:]
            for i, ent in enumerate(remaining):
                insert_pos = min(
                    int((i + 1) * len(result) / (len(remaining) + 1)),
                    len(result)
                )
                result.insert(insert_pos, ent)
            flat_entities = result
        else:
            flat_entities = sorted_entities

    order_description = rule["rule"]
    entity_order_str = " → ".join([f"{e['type']}({e['text']})" for e in flat_entities])
    return flat_entities, f"{order_description}. Actual order: {entity_order_str}"


def get_replacement_entity(entities, etype, exclude_texts):
    if etype not in entities or not entities[etype]:
        return None
    candidates = [e for e in entities[etype] if e not in exclude_texts]
    if not candidates:
        candidates = entities[etype]
    return random.choice(candidates)


# ============================================================
# 策略 1：实体替换 - 0%（保留代码，但权重为0）
# ============================================================
def strategy_entity_substitution(original_data, entities, llm_kwargs):
    item = random.choice(original_data)
    text = item['text']
    old_entities = item.get('entities', [])
    if not old_entities:
        return None

    replace_ratio = random.uniform(0.3, 0.8)
    num_to_replace = max(1, int(len(old_entities) * replace_ratio))
    indices_to_replace = set(random.sample(range(len(old_entities)), min(num_to_replace, len(old_entities))))

    replacements = {}
    for idx in indices_to_replace:
        ent = old_entities[idx]
        etype = ent.get('entity_type', '')
        old_text = ent.get('text', '')
        new_text = get_replacement_entity(entities, etype, {old_text})
        if new_text is None:
            new_text = old_text
        replacements[idx] = {'old': old_text, 'new': new_text, 'type': etype,
                             'old_start': ent['start'], 'old_end': ent['end']}

    new_text = text
    for idx in sorted(indices_to_replace, key=lambda i: old_entities[i]['start'], reverse=True):
        rep = replacements[idx]
        new_text = new_text[:rep['old_start']] + rep['new'] + new_text[rep['old_end']:]

    final_entities = []
    used_positions = set()
    for i, ent in enumerate(old_entities):
        etype = ent.get('entity_type', '')
        entity_text = replacements[i]['new'] if i in replacements else ent.get('text', '')
        start = new_text.find(entity_text)
        while start != -1 and start in used_positions:
            start = new_text.find(entity_text, start + 1)
        if start != -1:
            used_positions.add(start)
            final_entities.append({
                "start": start, "end": start + len(entity_text),
                "entity_type": etype, "text": entity_text
            })

    if not final_entities or len(new_text) < TARGET_TEXT_LENGTH_MIN or len(new_text) > TARGET_TEXT_LENGTH_MAX or new_text == text:
        return None

    return {"text": new_text, "entities": final_entities, "strategy": "entity_substitution"}


# ============================================================
# 策略 2：原文改写 - 50%
# ============================================================
def strategy_paraphrase(original_data, entities, llm_kwargs):
    item = random.choice(original_data)
    text = item['text']
    old_entities = item.get('entities', [])
    if not old_entities or len(text) < 50:
        return None

    entity_replacements = {}
    new_entity_list = []
    for ent in old_entities:
        etype = ent.get('entity_type', '')
        old_text = ent.get('text', '')
        new_text = get_replacement_entity(entities, etype, {old_text})
        if new_text is None:
            new_text = old_text
        entity_replacements[old_text] = new_text
        new_entity_list.append(f"[{etype}]{new_text}[/{etype}]")

    new_entity_str = "\n".join([f"   - {e}" for e in new_entity_list])
    instruction = random.choice(PARAPHRASE_INSTRUCTIONS)
    target_len = compute_target_length(len(old_entities))

    prompt = f"""You are a paleontology text rewriting assistant. Rewrite the following academic text according to the specified instruction, using DIFFERENT entity names than the original.

Original text:
{text}

Original entities (DO NOT use these, use the replacements below):
{', '.join([f'[{e.get("entity_type", "")}]{e.get("text", "")}' for e in old_entities])}

Replacement entities (use these INSTEAD of the original ones):
{new_entity_str}

Rewriting instruction: {instruction}

Requirements:
1. Use [ENTITY_TYPE]entity_text[/ENTITY_TYPE] tags for all entities
2. Use ONLY the replacement entities listed above — do NOT reuse any original entity text
3. Follow the rewriting instruction strictly: {instruction}
4. The rewritten text should be {target_len[0]}-{target_len[1]} characters (excluding tags)
5. The rewritten text MUST differ substantially from the original in wording and sentence structure
6. Maintain formal academic English appropriate for paleontology
7. Output ONLY the rewritten text with entity tags, nothing else

Rewritten Text:"""

    raw = call_llm(prompt, **llm_kwargs)
    if not raw:
        return None

    clean_text, parsed_entities = parse_entities_from_text(raw)
    if not clean_text or not parsed_entities:
        return None
    if len(clean_text) < 30 or len(clean_text) > 1200:
        return None
    if clean_text == text:
        return None

    return {"text": clean_text, "entities": parsed_entities, "strategy": "paraphrase"}


# ============================================================
# 策略 3：实体组合 + 风格生成 - 50%
# ============================================================
def strategy_entity_combination(original_data, entities, llm_kwargs):
    entity_types = select_entity_types_weighted()
    sampled = sample_entities_for_types(entities, entity_types, max_per_type=5)
    if not sampled:
        return None

    rule = random.choice(ENTITY_ORDER_RULES)
    ordered_entity_list, order_description = order_entities_by_rule(sampled, rule)
    if not ordered_entity_list:
        return None

    style = random.choice(GENERATION_STYLES)

    ordered_str = "\n".join([
        f"   {i+1}. [{e['type']}]{e['text']}[/{e['type']}] ({ENTITY_TYPE_DESCRIPTIONS.get(e['type'], e['type'])})"
        for i, e in enumerate(ordered_entity_list)
    ])

    type_counts = {}
    for e in ordered_entity_list:
        type_counts[e['type']] = type_counts.get(e['type'], 0) + 1
    type_summary = ", ".join([f"{t}×{c}" for t, c in type_counts.items()])
    target_len = compute_target_length(len(ordered_entity_list))

    prompt = f"""Generate a short paleontology academic text (1-3 sentences) using the following entities in the specified order and style.

Entity arrangement rule: {order_description}
Entity composition: {type_summary}

Entities (appear in this order):
{ordered_str}

Generation style: {style}

Entity type definitions:
   - taxa: biological species names (genus, species, abbreviated forms)
   - location: geographic place names (countries, regions, basins)
   - section: named geological sections/outcrops
   - strata: lithostratigraphic units (Formation, Group, Member)
   - lithology: rock type descriptions
   - facies: depositional/sedimentary facies
   - age: geological time terms (periods, epochs, ages)

CRITICAL RULES:
1. Entities MUST appear in the specified order: {' → '.join([f"{e['type']}({e['text']})" for e in ordered_entity_list])}
2. Follow the generation style strictly: {style}
3. Text length: {target_len[0]}-{target_len[1]} characters (excluding tags)
4. DO NOT start with "The presence of", "Analysis of", or "Fossil remains of"
5. Use ALL listed entities with exact [TYPE]text[/TYPE] tags
6. DO NOT add suffixes to entity text
7. Ensure the text sounds natural and academically appropriate for paleontology
8. Output ONLY the generated text with entity tags, nothing else

Generated Text:"""

    raw = call_llm(prompt, **llm_kwargs)
    if not raw:
        return None

    clean_text, parsed_entities = parse_entities_from_text(raw)
    if not clean_text or not parsed_entities:
        return None
    if len(clean_text) < TARGET_TEXT_LENGTH_MIN or len(clean_text) > TARGET_TEXT_LENGTH_MAX:
        return None

    return {"text": clean_text, "entities": parsed_entities, "strategy": "entity_combination"}


# ============================================================
# 测试模式
# ============================================================
def run_test(original_data, entities, llm_kwargs):
    print("\n" + "=" * 70, flush=True)
    print("  测试模式 —— 验证 API 连通性和生成效果", flush=True)
    print("=" * 70, flush=True)

    # 1. API 连通性
    print(f"\n--- [1] API 连通性测试 ---", flush=True)
    r = call_llm("Respond with: TEST_OK", temperature=0.0, max_tokens=64, **llm_kwargs)
    if r is None:
        print("✗ API 连通性测试失败！", flush=True)
        return False
    print(f"✓ API 连通正常，响应: {r[:100]}", flush=True)

    # 2. 策略2测试（改写）
    print(f"\n--- [2] 策略2（原文改写）测试 ---", flush=True)
    result2 = strategy_paraphrase(original_data, entities, llm_kwargs)
    if result2:
        print(f"✓ 改写成功: 文本 {len(result2['text'])} 字符, {len(result2['entities'])} 个实体", flush=True)
        print(f"  文本: {result2['text'][:150]}...", flush=True)
        for ent in result2['entities'][:5]:
            print(f"    [{ent['entity_type']}] \"{ent['text']}\"", flush=True)
    else:
        print("△ 改写返回 None（可能是随机数据不合适，不影响整体）", flush=True)

    # 3. 策略3测试（组合生成）
    print(f"\n--- [3] 策略3（实体组合+风格生成）测试 ---", flush=True)
    result3 = strategy_entity_combination(original_data, entities, llm_kwargs)
    if result3:
        print(f"✓ 组合生成成功: 文本 {len(result3['text'])} 字符, {len(result3['text'])} 个实体", flush=True)
        print(f"  文本: {result3['text'][:150]}...", flush=True)
        for ent in result3['entities'][:5]:
            print(f"    [{ent['entity_type']}] \"{ent['text']}\"", flush=True)
    else:
        print("△ 组合生成返回 None", flush=True)

    # 4. 多次测试统计成功率
    print(f"\n--- [4] 连续生成测试 (5次) ---", flush=True)
    ok2, ok3 = 0, 0
    for i in range(5):
        r2 = strategy_paraphrase(original_data, entities, llm_kwargs)
        r3 = strategy_entity_combination(original_data, entities, llm_kwargs)
        if r2: ok2 += 1
        if r3: ok3 += 1
        print(f"  第{i+1}次: 改写={'OK' if r2 else 'FAIL'}, 组合={'OK' if r3 else 'FAIL'}", flush=True)

    print(f"\n--- [5] 测试总结 ---", flush=True)
    print(f"  API 连通: ✓", flush=True)
    print(f"  策略2成功率: {ok2}/5", flush=True)
    print(f"  策略3成功率: {ok3}/5", flush=True)
    if ok2 + ok3 == 0:
        print("✗ 两种策略全部失败，请检查 API 或数据", flush=True)
        return False
    print("✓ 测试通过，可以开始正式增广", flush=True)
    print("=" * 70, flush=True)
    return True


# ============================================================
# 主流程
# ============================================================
