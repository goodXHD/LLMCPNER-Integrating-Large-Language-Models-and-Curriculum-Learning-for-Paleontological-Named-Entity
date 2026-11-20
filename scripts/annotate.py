#!/usr/bin/env python3
"""
GPT-NER 摘要标注系统（一次性抽取所有实体类型）
================================================
不分句、不分类型，整条摘要一次性送给 LLM 抽取所有实体，
输出 JSON 格式（含类型标签）。

运行: conda activate unsloth && python annotate_abstracts_only.py
"""

import os, sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ.update({"OPENBLAS_NUM_THREADS":"32","MKL_NUM_THREADS":"32","OMP_NUM_THREADS":"32"})

import json, re, time, argparse, signal
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, warnings
warnings.filterwarnings('ignore')

# ======================== 实体类型定义 ========================
ENTITY_TYPES = {
    "taxa": {
        "description": "Biological taxonomic names at ANY rank (genus, species, family, order, class, phylum, etc.), including open nomenclature forms with qualifiers like 'cf.', 'aff.', '?', 'sp.', 'sp. nov.', 'gen. nov.', 'subsp.', 'var.', 'spp.'. Extract the FULL taxonomic name as a single entity — do NOT split genus and species apart.",
        "examples": "Homo sapiens, T. rex, Ammonoidea, cf. Trigonia, Ammonites aff. amalthei, Pecten? sp., Ostrea sp. nov.",
    },
    "age": {
        "description": "Geological time periods, ages, epochs, or absolute dates.",
        "examples": "Cambrian, Late Cretaceous, 230Th dated, 8,736 ± 56 yr BP, Ma, kyr BP, Holocene, Pleistocene, Middle Devonian",
    },
    "location": {
        "description": "Named geographic locations. Each location name is a SEPARATE entity. A single entity should be a contiguous place name where the words form one proper noun.",
        "examples": "South Wales, Lutao, Mochras, Mediterranean, Tethys, Panthalassa, South China Block",
    },
    "strata": {
        "description": "Named geological stratigraphic units that EXPLICITLY include a stratigraphic suffix (Formation, Group, Member, Stage, Unit, Bed, or abbreviations Fm., Grp., Mbr.). A name WITHOUT such suffix is NOT strata.",
        "examples": "Morrison Formation, Chattanooga Shale Group, Redwall Limestone Member, Emeishan Basalt Formation",
    },
    "lithology": {
        "description": "Rock types or sediment descriptions.",
        "examples": "limestone, sandstone, shale, mudstone, dolomite, chert, conglomerate, siltstone, marl, basalt, granite, tuff, breccia",
    },
    "facies": {
        "description": "Geological facies descriptions or names, typically including the word 'facies'.",
        "examples": "reef facies, basinal facies, transgressive facies, shallow-marine facies, turbiditic facies",
    },
    "section": {
        "description": "Named geological sections that EXPLICITLY include the word 'Section' (or 'Sec.') in their name. A place name without 'Section' is NOT a section entity.",
        "examples": "Meishan Section, GSSP section, Section A, Wagnong Section, D-C Section",
    },
}

# ======================== 配置 ========================
DEFAULT_INPUT = os.environ.get("PALEO_DEFAULT_INPUT", "data/cluster_abstracts.json")
DEFAULT_OUTPUT_DIR = os.environ.get("PALEO_DEFAULT_OUTPUT_DIR", "outputs/annotation")
DEFAULT_FEWSHOT = os.environ.get("PALEO_DEFAULT_FEWSHOT", "data/fewshot.json")
DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")
MAX_ABSTRACT_CHARS = 10000
CONFIDENCE_THRESHOLD = 80
MAX_CONCURRENT = 50
RETRY_TIMES = 5
FEWSHOT_MAX_CHARS = 800  # fewshot 示例文本最大字符数（太长的不选）
FEWSHOT_NUM = 3          # 选取 fewshot 示例数量

# ======================== Fewshot 加载 ========================
def load_fewshot_examples(fewshot_path: str, max_chars: int = FEWSHOT_MAX_CHARS, num: int = FEWSHOT_NUM) -> List[Dict]:
    """从 merged_data.json 加载 fewshot 示例，选择类型覆盖最多、长度合适的样本。"""
    with open(fewshot_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # 过滤：有实体且文本不过长
    candidates = [d for d in all_data if d.get("entities") and len(d.get("text", "")) <= max_chars]
    if not candidates:
        print(f"  警告: fewshot 文件中无合适样本", flush=True)
        return []

    # 按类型覆盖广度排序，优先选覆盖类型最多的
    def type_coverage(d):
        return len(set(e["entity_type"] for e in d["entities"]))

    candidates.sort(key=type_coverage, reverse=True)

    # 贪心选择：每轮选与已选样本类型覆盖互补最多的
    selected = []
    covered_types = set()

    for _ in range(min(num, len(candidates))):
        best = None
        best_score = -1
        for c in candidates:
            if c in selected:
                continue
            new_types = set(e["entity_type"] for e in c["entities"]) - covered_types
            score = len(new_types) * 100 + type_coverage(c)  # 互补性优先
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            break
        selected.append(best)
        covered_types.update(e["entity_type"] for e in best["entities"])

    # 转换为 prompt 格式
    fewshot_formatted = []
    for d in selected:
        entities_out = [{"text": e["text"], "type": e["entity_type"]} for e in d["entities"]]
        fewshot_formatted.append({"text": d["text"], "entities": entities_out})

    type_list = sorted(covered_types)
    print(f"  加载 fewshot: {len(fewshot_formatted)} 条样本，覆盖类型: {type_list}", flush=True)
    return fewshot_formatted


# ======================== Prompt 构建 ========================
def build_extraction_prompt(abstract_text: str, fewshot_examples: List[Dict] = None) -> Tuple[str, str]:
    type_descs = []
    for etype, info in ENTITY_TYPES.items():
        type_descs.append(f'  "{etype}": {info["description"]}\n    Examples: {info["examples"]}')

    system_prompt = (
        "You are an expert paleontology NER system. "
        "Extract ALL entities from the given abstract that belong to any of the defined entity types. "
        "Output strictly in JSON format."
    )

    # 构建 fewshot 部分
    fewshot_section = ""
    if fewshot_examples:
        parts = []
        for i, ex in enumerate(fewshot_examples):
            ents_json = json.dumps(ex["entities"], ensure_ascii=False)
            parts.append(f"Example {i+1}:\nAbstract: {ex['text']}\nOutput: {json.dumps({'entities': ex['entities']}, ensure_ascii=False)}")
        fewshot_section = "\n\nHere are some examples:\n\n" + "\n\n".join(parts) + "\n\n---\n"

    user_prompt = f"""Extract all entities from the following abstract that belong to these entity types:

{chr(10).join(type_descs)}
{fewshot_section}
RULES:
- Extract each entity as the EXACT text span from the abstract.
- Each entity must have a "type" field indicating its entity type.
- If an entity could belong to multiple types, choose the MOST SPECIFIC one.
- Do NOT invent entities not present in the text.
- Do NOT split multi-word entities (e.g., "Homo sapiens" is ONE entity).

Output strictly in JSON:
{{"entities": [{{"text": "entity text", "type": "entity_type"}}, ...]}}

If no entities found: {{"entities": []}}

Abstract:
{abstract_text}"""
    return system_prompt, user_prompt


def build_verification_prompt(abstract_text: str, entity_text: str, entity_type: str) -> Tuple[str, str]:
    info = ENTITY_TYPES.get(entity_type, {"description": f"{entity_type} entities", "examples": ""})
    system_prompt = "You are an expert paleontology NER verification system. Output strictly in JSON format."
    user_prompt = f"""Abstract: {abstract_text}

Is "{entity_text}" a valid "{entity_type}" entity?
Definition: {info['description']}
Examples: {info['examples']}

JSON output:
{{"is_valid": true/false, "confidence": 85, "reason": "brief explanation"}}"""
    return system_prompt, user_prompt


# ======================== JSON 解析 ========================
def parse_extraction_response(response: str) -> List[Dict]:
    """解析 LLM 返回的 JSON 格式实体列表。"""
    # 尝试直接解析
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "entities" in data:
            ents = data["entities"]
            if isinstance(ents, list):
                return [e for e in ents if isinstance(e, dict) and "text" in e and "type" in e]
    except json.JSONDecodeError:
        pass
    # 尝试从文本中提取 JSON
    for pattern in [r'\{[\s\S]*"entities"[\s\S]*\}', r'```(?:json)?\s*([\s\S]*?)```']:
        m = re.search(pattern, response)
        if m:
            try:
                txt = m.group(1) if m.lastindex else m.group()
                data = json.loads(txt.strip())
                ents = data.get("entities", [])
                if isinstance(ents, list):
                    return [e for e in ents if isinstance(e, dict) and "text" in e and "type" in e]
            except (json.JSONDecodeError, AttributeError):
                pass
    return []


# ======================== LLM API 调用 ========================
_conn_error_counter = {"value": 0, "lock": threading.Lock()}
_stop_event = threading.Event()

def _increment_conn_errors():
    with _conn_error_counter["lock"]: _conn_error_counter["value"] += 1
def _get_conn_errors():
    with _conn_error_counter["lock"]: return _conn_error_counter["value"]
def should_stop(): return _stop_event.is_set()
def request_stop(reason: str):
    print(f"\n!!! 停止请求: {reason} !!!", flush=True)
    _stop_event.set()


# ======================== 全局 OpenAI 客户端复用 ========================
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


def call_llm(prompt, system_prompt="", api_base=DEFAULT_API_BASE, api_key=DEFAULT_API_KEY,
             model=DEFAULT_MODEL, temperature=0.0, max_tokens=4096):
    if should_stop(): return None, False
    try:
        client = _get_client(api_base, api_key)
        msgs = []
        if system_prompt: msgs.append({"role":"system","content":system_prompt})
        msgs.append({"role":"user","content":prompt})
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content.strip(), False
    except Exception as e:
        err = str(e).lower()
        is_conn = any(kw in err for kw in ['connection','connect','refused','timeout','timed out'])
        is_quota = any(kw in err for kw in ['quota','insufficient_quota','billing','429','402','rate_limit','capacity','credit','balance'])
        if is_quota: request_stop(f"API 余额不足或限额 (错误: {e})")
        if is_conn: _increment_conn_errors()
        # 连接错误时重置客户端
        if is_conn:
            global _global_client
            with _client_lock:
                _global_client = None
        print(f"  API 调用失败: {e}", flush=True)
        return None, is_conn


def call_llm_with_retry(prompt, system_prompt="", **kwargs):
    conn_backoff = [5,15,30,60,120]
    other_backoff = [2,4,8,16,32]
    for attempt in range(RETRY_TIMES):
        if should_stop(): return None
        result, is_conn = call_llm(prompt, system_prompt=system_prompt, **kwargs)
        if result is not None: return result
        if should_stop(): return None
        if attempt < RETRY_TIMES - 1:
            bl = conn_backoff if is_conn else other_backoff
            wait = bl[min(attempt, len(bl)-1)]
            print(f"    {'连接错误' if is_conn else 'API错误'}，{wait}s 后重试 ({attempt+1}/{RETRY_TIMES})...", flush=True)
            time.sleep(wait)
    print(f"    API 调用失败，已重试 {RETRY_TIMES} 次", flush=True)
    return None


# ======================== 标注核心逻辑 ========================
def annotate_one_abstract(abstract_text, llm_kwargs, verify=True, fewshot_examples=None):
    """对单条摘要一次性抽取所有实体类型，可选验证。"""
    # Step-1: 一次性提取
    sys_p, usr_p = build_extraction_prompt(abstract_text, fewshot_examples=fewshot_examples)
    response = call_llm_with_retry(usr_p, system_prompt=sys_p, **llm_kwargs)
    if response is None:
        return {"entities":[], "n_extracted":0, "n_verified":0, "status":"api_error", "raw_response":""}

    raw_entities = parse_extraction_response(response)
    n_extracted = len(raw_entities)

    # 过滤合法类型
    valid_types = set(ENTITY_TYPES.keys())
    raw_entities = [e for e in raw_entities if e.get("type") in valid_types]

    # Step-2: 可选验证
    final_entities = []
    n_verified = 0
    has_verify_error = False

    if verify and raw_entities:
        for ent in raw_entities:
            if should_stop():
                has_verify_error = True
                break
            v_sys, v_usr = build_verification_prompt(abstract_text, ent["text"], ent["type"])
            v_resp = call_llm_with_retry(v_usr, system_prompt=v_sys, **llm_kwargs)
            if v_resp is None:
                has_verify_error = True
                continue
            try:
                m = re.search(r'\{[^{}]*\}', v_resp)
                if m:
                    d = json.loads(m.group())
                    is_valid = d.get("is_valid", False)
                    confidence = int(d.get("confidence", 0))
                    reason = d.get("reason", "")
                else:
                    is_valid = "yes" in v_resp.lower() and "no" not in v_resp.lower()[:10]
                    confidence = 50 if is_valid else 0
                    reason = v_resp[:100]
            except (json.JSONDecodeError, ValueError):
                is_valid = "yes" in v_resp.lower()
                confidence = 50 if is_valid else 0
                reason = v_resp[:100]
            if is_valid:
                n_verified += 1
                final_entities.append({"text":ent["text"],"type":ent["type"],"confidence":confidence,"reason":reason})
    else:
        for ent in raw_entities:
            final_entities.append({"text":ent["text"],"type":ent["type"],"confidence":100,"reason":"not verified"})
        n_verified = len(raw_entities)

    result = {"entities":final_entities, "n_extracted":n_extracted, "n_verified":n_verified,
              "raw_response":response[:500] if response else ""}
    if has_verify_error: result["status"] = "partial_error"
    return result


# ======================== 测试模式 ========================
def run_test(abstracts, llm_kwargs, verify=True, fewshot_examples=None):
    print("\n" + "="*70, flush=True)
    print("  测试模式 —— 验证 API 连通性和标注效果", flush=True)
    print("="*70, flush=True)

    test_abstract = None
    for a in abstracts:
        text = a.get("abstract","").strip()
        if text and 50 < len(text) < 2000:
            test_abstract = a
            break
    if test_abstract is None:
        print("✗ 未找到合适的测试摘要", flush=True)
        return False

    abstract_text = test_abstract["abstract"].strip()
    paper_id = test_abstract.get("paper_id","unknown")
    print(f"\n测试摘要 (paper_id={paper_id}, {len(abstract_text)} 字符):", flush=True)
    print(f"  {abstract_text[:200]}{'...' if len(abstract_text)>200 else ''}", flush=True)

    # API 连通性
    print(f"\n--- [1] API 连通性测试 ---", flush=True)
    r, _ = call_llm("Respond with: TEST_OK", system_prompt="Test", max_tokens=64, **llm_kwargs)
    if r is None:
        print("✗ API 连通性测试失败！", flush=True)
        return False
    print(f"✓ API 连通正常，响应: {r[:100]}", flush=True)

    # 抽取测试
    print(f"\n--- [2] 一次性抽取所有实体类型 (fewshot: {len(fewshot_examples) if fewshot_examples else 0} 条) ---", flush=True)
    result = annotate_one_abstract(abstract_text, llm_kwargs, verify=verify, fewshot_examples=fewshot_examples)
    if result.get("status") == "api_error":
        print("✗ 提取失败", flush=True)
        return False

    print(f"  提取: {result['n_extracted']} 个 | 验证通过: {result['n_verified']} 个", flush=True)

    by_type = defaultdict(list)
    for ent in result["entities"]:
        by_type[ent["type"]].append(ent)

    for etype in ENTITY_TYPES:
        ents = by_type.get(etype, [])
        if ents:
            print(f"\n  [{etype}] ({len(ents)} 个):", flush=True)
            for ent in ents:
                conf = ent["confidence"]
                mark = "✓" if conf >= CONFIDENCE_THRESHOLD else "△"
                print(f"    {mark} \"{ent['text']}\" conf={conf} {ent.get('reason','')}", flush=True)
        else:
            print(f"\n  [{etype}] （无）", flush=True)

    print(f"\n--- [3] LLM 原始输出片段 ---", flush=True)
    print(f"  {result.get('raw_response','')[:300]}", flush=True)

    print(f"\n--- [4] 测试总结 ---", flush=True)
    print(f"✓ 标注测试通过！发现 {len(result['entities'])} 个实体，覆盖 {len(by_type)} 种类型", flush=True)
    print("="*70, flush=True)
    return True


# ======================== 批量标注 ========================
def annotate_batch(abstracts, llm_kwargs, max_concurrent=MAX_CONCURRENT,
                   output_dir=DEFAULT_OUTPUT_DIR, verify=True, fewshot_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")

    # 加载断点
    completed_ids = set()
    results = []
    total_done_before = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
            results = ckpt.get("results", [])
            completed_ids = set(ckpt.get("completed_ids", []))
            total_done_before = len(completed_ids)
        print(f"\n从断点恢复: 已完成 {len(completed_ids)}/{len(abstracts)} 条", flush=True)

    # 过滤过长摘要
    valid_abstracts = []
    skipped = []
    for a in abstracts:
        aid = a.get("paper_id", str(id(a)))
        text = a.get("abstract","").strip()
        if not text: skipped.append(aid); continue
        if len(text) > MAX_ABSTRACT_CHARS:
            print(f"  [{aid}] 摘要过长 ({len(text)} 字符)，跳过", flush=True)
            skipped.append(aid); continue
        valid_abstracts.append(a)

    todo = [a for a in valid_abstracts if a.get("paper_id", str(id(a))) not in completed_ids]
    total_all = len(valid_abstracts)
    total_todo = len(todo)
    if not todo:
        print("所有摘要已标注完成，无需处理。", flush=True)
        return results

    print(f"\n总摘要: {len(abstracts)} | 过长跳过: {len(skipped)} | 有效: {total_all} | 已完成: {total_done_before} | 待标注: {total_todo}", flush=True)
    print(f"并发数: {max_concurrent} | 验证: {'开启' if verify else '关闭'}", flush=True)

    all_tasks = [(a.get("paper_id", str(id(a))), a.get("abstract","").strip()) for a in todo]
    total_tasks = len(all_tasks)
    print(f"总任务数: {total_tasks} (每条摘要 1 次抽取调用)", flush=True)

    # 断点保存
    _ckpt_lock = threading.Lock()
    def _save_checkpoint():
        with _ckpt_lock:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"results":results,"completed_ids":list(completed_ids),"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False)
            with open(os.path.join(output_dir, "annotated_abstracts_interim.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 信号处理
    _orig_sigint = signal.getsignal(signal.SIGINT)
    _orig_sigterm = signal.getsignal(signal.SIGTERM)
    def _sig_handler(signum, frame):
        sn = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n!!! 收到 {sn} 信号，正在保存断点并停止... !!!", flush=True)
        request_stop(f"收到 {sn} 信号")
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    _lock = threading.Lock()
    _stats = {"total_entities":0, "total_high_conf":0, "completed_abstracts":0, "completed_tasks":0, "errors":0}

    with _conn_error_counter["lock"]: _conn_error_counter["value"] = 0
    _stop_event.clear()
    t0 = time.time()

    # 心跳
    _stop_hb = threading.Event()
    def _heartbeat():
        while not _stop_hb.is_set():
            _stop_hb.wait(30)
            if _stop_hb.is_set(): break
            with _lock:
                ct,cp,te = _stats["completed_tasks"], _stats["completed_abstracts"], _stats["total_entities"]
                elapsed = time.time()-t0
                speed = ct/elapsed if elapsed>0 else 0
                eta = (total_tasks-ct)/speed if speed>0 else 0
                eta_s = f"{int(eta//60)}m{int(eta%60)}s" if speed>0 else "N/A"
            st = " [STOPPING]" if should_stop() else ""
            print(f"[进度] 摘要 {cp}/{total_all}, 任务 {ct}/{total_tasks}, 实体 {te} | {elapsed:.0f}s {speed:.1f}task/s ETA:{eta_s}{st}", flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_map = {}
        for pid, atext in all_tasks:
            future_map[executor.submit(annotate_one_abstract, atext, llm_kwargs, verify, fewshot_examples)] = (pid, atext)

        for future in as_completed(future_map):
            pid, atext = future_map[future]
            if should_stop(): continue
            try:
                task_result = future.result()
                with _lock:
                    _stats["completed_tasks"] += 1
                    entities = task_result["entities"]
                    _stats["total_entities"] += len(entities)
                    _stats["total_high_conf"] += sum(1 for e in entities if e.get("confidence",0)>=CONFIDENCE_THRESHOLD)
                    if task_result.get("status") in ("api_error","partial_error"):
                        _stats["errors"] += 1

                    # 按类型分组
                    by_type = defaultdict(list)
                    for ent in entities:
                        by_type[ent["type"]].append(ent)
                    annotations = {}
                    for etype in ENTITY_TYPES:
                        ents = by_type.get(etype, [])
                        annotations[etype] = {"entities":ents, "total_extracted":len(ents), "total_high_conf":len(ents)}

                    all_confs = [e["confidence"] for e in entities]
                    avg_conf = np.mean(all_confs) if all_confs else 0

                    results.append({
                        "paper_id": pid,
                        "abstract": atext,
                        "annotations": annotations,
                        "paper_avg_confidence": avg_conf,
                    })
                    completed_ids.add(pid)
                    _stats["completed_abstracts"] += 1

                    n_ents = len(entities)
                    n_high = _stats["total_high_conf"]
                    done = total_done_before + _stats["completed_abstracts"]
                    elapsed = time.time()-t0
                    speed = _stats["completed_abstracts"]/elapsed if elapsed>0 else 0
                    eta_s = f"{int((total_todo-_stats['completed_abstracts'])/speed//60) if speed>0 else '?'}m"
                    cm = "OK" if avg_conf>=CONFIDENCE_THRESHOLD else "LOW"
                    print(f"[{done}/{total_all}] ({done*100/total_all:.1f}%) {pid} | {n_ents}实体 conf={avg_conf:.0f}{cm} | {elapsed:.0f}s {speed:.2f}条/s ETA:{eta_s}", flush=True)
                    _save_checkpoint()
            except Exception as e:
                with _lock: _stats["errors"] += 1; _stats["completed_tasks"] += 1
                print(f"  ERROR [{pid}]: {e}", flush=True)

    _stop_hb.set()
    signal.signal(signal.SIGINT, _orig_sigint)
    signal.signal(signal.SIGTERM, _orig_sigterm)
    _save_checkpoint()

    t1 = time.time()
    sr = " (因余额不足/限额/信号而提前停止)" if should_stop() else ""
    print(f"\n标注完成{sr}: {_stats['completed_abstracts']} 条, 耗时 {t1-t0:.1f}s, 错误 {_stats['errors']} 任务", flush=True)
    print(f"总实体: {_stats['total_entities']}, 高置信(>={CONFIDENCE_THRESHOLD}): {_stats['total_high_conf']}", flush=True)
    if should_stop():
        print(f"\n!!! 已保存 {_stats['completed_abstracts']} 条结果到断点文件 !!!", flush=True)
        print(f"  重新运行将自动从断点恢复", flush=True)
    return results


# ======================== 结果保存 ========================
def compute_paper_confidence(result):
    annotations = result.get("annotations", {})
    confs = [e["confidence"] for ann in annotations.values() for e in ann.get("entities", [])]
    return np.mean(confs) if confs else 0.0

def save_final_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for r in results:
        if "annotations" in r: r["paper_avg_confidence"] = compute_paper_confidence(r)

    # 1. 完整结果
    p = os.path.join(output_dir, "annotated_abstracts_full.json")
    with open(p,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"完整结果: {len(results)} 条 -> {p}", flush=True)

    # 2. 高置信度
    hc = [r for r in results if r.get("paper_avg_confidence",0)>=CONFIDENCE_THRESHOLD and any(ann.get("entities") for ann in r.get("annotations",{}).values())]
    p2 = os.path.join(output_dir, "annotated_abstracts_high_confidence.json")
    with open(p2,"w",encoding="utf-8") as f: json.dump(hc,f,ensure_ascii=False,indent=2)
    print(f"高置信度: {len(hc)} 条 -> {p2}", flush=True)

    # 3. 训练格式
    training_data = []
    for r in results:
        if r.get("paper_avg_confidence",0) < CONFIDENCE_THRESHOLD: continue
        text = r.get("abstract","")
        all_ents = []
        # 记录已占用的 span，避免同一位置被重复匹配
        occupied_spans = []  # [(start, end), ...]

        for etype, ann in r.get("annotations",{}).items():
            for ent in ann.get("entities",[]):
                ent_text = ent["text"]
                ent_len = len(ent_text)
                start = -1

                # 精确匹配：找第一个未被占用的出现位置
                search_from = 0
                while search_from <= len(text) - ent_len:
                    pos = text.find(ent_text, search_from)
                    if pos == -1:
                        break
                    # 检查该位置是否已被占用
                    overlap = any(not (pos + ent_len <= s or pos >= e) for s, e in occupied_spans)
                    if not overlap:
                        start = pos
                        break
                    search_from = pos + 1

                # 精确匹配失败，尝试模糊匹配（忽略大小写/空格差异）
                if start == -1:
                    ent_lower = ent_text.strip().lower()
                    ent_clean = re.sub(r'\s+', ' ', ent_lower)
                    for i in range(len(text) - ent_len + 1):
                        seg = text[i:i+ent_len]
                        seg_clean = re.sub(r'\s+', ' ', seg.strip().lower())
                        if seg_clean == ent_clean:
                            overlap = any(not (i + ent_len <= s or i >= e) for s, e in occupied_spans)
                            if not overlap:
                                start = i
                                break

                if start >= 0:
                    occupied_spans.append((start, start + ent_len))
                    all_ents.append({"start":start,"end":start+ent_len,"entity_type":etype,"text":ent_text,"confidence":ent["confidence"]})

        if all_ents:
            training_data.append({"text":text,"entities":sorted(all_ents,key=lambda x:x["start"]),"source":r.get("paper_id",""),"data_type":"abstract_only"})
    p3 = os.path.join(output_dir, "training_data.json")
    with open(p3,"w",encoding="utf-8") as f: json.dump(training_data,f,ensure_ascii=False,indent=2)
    print(f"训练数据: {len(training_data)} 条 -> {p3}", flush=True)

    # 4. 统计
    stats = {"total_abstracts":len(results),"high_conf":len(hc),"training":len(training_data),"entity_type_stats":{}}
    for etype in ENTITY_TYPES:
        ents = [e for r in results for e in r.get("annotations",{}).get(etype,{}).get("entities",[])]
        stats["entity_type_stats"][etype] = {"total":len(ents),"avg_conf":np.mean([e["confidence"] for e in ents]) if ents else 0}
    p4 = os.path.join(output_dir, "annotation_stats.json")
    with open(p4,"w",encoding="utf-8") as f: json.dump(stats,f,ensure_ascii=False,indent=2)
    print(f"统计摘要 -> {p4}", flush=True)

    print("\n" + "="*60, flush=True)
    print(f"总摘要: {stats['total_abstracts']} | 高置信: {stats['high_conf']} | 训练样本: {stats['training']}", flush=True)
    for etype, s in stats["entity_type_stats"].items():
        print(f"  {etype}: {s['total']} 个, 平均置信度 {s['avg_conf']:.1f}", flush=True)


# ======================== 主函数 ========================
def main():
    global CONFIDENCE_THRESHOLD
    parser = argparse.ArgumentParser(description="GPT-NER 摘要标注系统（一次性抽取所有类型）")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fewshot", default=DEFAULT_FEWSHOT, help="Fewshot 示例文件路径")
    parser.add_argument("--fewshot-num", type=int, default=FEWSHOT_NUM, help="Fewshot 示例数量")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-abstracts", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=int, default=80)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--no-verify", action="store_true", help="跳过验证步骤")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.confidence_threshold
    verify = not args.no_verify
    llm_kwargs = {"api_base":args.api_base, "api_key":args.api_key, "model":args.model}

    print("="*60, flush=True)
    print("GPT-NER 摘要标注系统（一次性抽取所有类型）", flush=True)
    print("="*60, flush=True)
    print(f"输入: {args.input}", flush=True)
    print(f"输出: {args.output_dir}", flush=True)
    print(f"Fewshot: {args.fewshot} (选取 {args.fewshot_num} 条)", flush=True)
    print(f"API: {args.api_base} | 模型: {args.model}", flush=True)
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD} | 验证: {verify}", flush=True)

    # 加载 fewshot 示例
    print(f"\n加载 fewshot 示例...", flush=True)
    fewshot_examples = load_fewshot_examples(args.fewshot, num=args.fewshot_num)

    with open(args.input, "r", encoding="utf-8") as f:
        abstracts = json.load(f)
    print(f"加载摘要: {len(abstracts)} 条", flush=True)
    if args.max_abstracts:
        abstracts = abstracts[:args.max_abstracts]
        print(f"限制: 仅处理前 {args.max_abstracts} 条", flush=True)

    if args.test:
        sys.exit(0 if run_test(abstracts, llm_kwargs, verify, fewshot_examples=fewshot_examples) else 1)

    if not args.skip_test:
        print("\n--- 自动快速连通性检查 ---", flush=True)
        r, _ = call_llm("Respond with: OK", system_prompt="Test", max_tokens=64, **llm_kwargs)
        if r is None:
            print("✗ API 连通性检查失败！请使用 --test 运行详细测试", flush=True)
            sys.exit(1)
        print("✓ API 连通正常，开始标注...", flush=True)

    results = annotate_batch(abstracts, llm_kwargs, max_concurrent=args.max_concurrent,
                             output_dir=args.output_dir, verify=verify, fewshot_examples=fewshot_examples)
    save_final_results(results, args.output_dir)

    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if not should_stop() and os.path.exists(ckpt):
        os.remove(ckpt)
        print(f"断点已清理: {ckpt}", flush=True)

    if should_stop():
        print(f"\n!!! 程序提前停止，已保存 {len(results)} 条结果，重新运行可从断点恢复 !!!", flush=True)
    else:
        print(f"\n全部完成！结果保存在: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
