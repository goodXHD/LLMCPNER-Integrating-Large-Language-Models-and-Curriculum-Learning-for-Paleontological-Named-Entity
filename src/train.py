#!/usr/bin/env python3
"""Train the complete LLMCPNER model and evaluate it once on the test set."""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import (BATCH_SIZE, DATA_DIR, ENTITY_TYPES, MAX_SPAN_LENGTH,
                    MODEL_PATH, NUM_LABELS, OUTPUT_DIR, TYPE_MAPPING)
from dataset import NERDataset, collate_fn
from models import SpanBERTForNER


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def normalize_samples(samples, keep_empty=False):
    normalized = []
    for sample in samples:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        entities = []
        for entity in sample.get("entities", []):
            label = entity.get("entity_type", entity.get("type", entity.get("label", "")))
            label = TYPE_MAPPING.get(label, label)
            try:
                start, end = int(entity["start"]), int(entity["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if label in ENTITY_TYPES and 0 <= start < end <= len(text):
                entities.append({"start": start, "end": end, "entity_type": label,
                                 "text": text[start:end]})
        if entities or keep_empty:
            normalized.append({**sample, "text": text, "entities": entities})
    return normalized


def setup_logger(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paleo_weak_ner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handlers = [logging.StreamHandler(),
                logging.FileHandler(output_dir / "train.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def weighted_forward(model, batch, device, weights=None):
    output = model.bert(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch["token_type_ids"].to(device))
    sequence = model.dropout(output.last_hidden_state)
    losses = []
    for i, (starts, ends, labels) in enumerate(zip(
            batch["span_starts"], batch["span_ends"], batch["span_labels"])):
        if starts.numel() == 0:
            continue
        embedding = model.span_embedding(sequence, i, starts.to(device), ends.to(device))
        loss = F.cross_entropy(model.classifier(embedding), labels.to(device), reduction="none")
        if weights is not None:
            loss = loss * weights[i].to(device)
        losses.append(loss.mean())
    return torch.stack(losses).mean()


def train_voters(train_data, tokenizer, output_dir, args, device, logger):
    voters = []
    sample_count = max(1, int(len(train_data) * args.voter_sample_ratio))
    for voter_id in range(args.num_voters):
        voter_seed = args.seed + voter_id * 1009
        rng = random.Random(voter_seed)
        sampled = [train_data[rng.randrange(len(train_data))] for _ in range(sample_count)]
        dataset = NERDataset(sampled, tokenizer)
        generator = torch.Generator().manual_seed(voter_seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, generator=generator)
        model = SpanBERTForNER(MODEL_PATH, NUM_LABELS, MAX_SPAN_LENGTH).to(device)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        steps = max(1, len(loader) * args.voter_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
        for epoch in range(args.voter_epochs):
            model.train()
            total = 0.0
            for batch in tqdm(loader, desc=f"Voter {voter_id + 1}/{args.num_voters}", leave=False):
                loss = weighted_forward(model, batch, device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total += loss.item()
            logger.info("Voter %d epoch %d loss %.4f", voter_id + 1, epoch + 1,
                        total / max(len(loader), 1))
        torch.save(model.state_dict(), output_dir / f"voter_{voter_id}.pt")
        model.eval()
        voters.append(model)
    return voters


def estimate_positive_confidence(train_data, tokenizer, voters, args, device):
    dataset = NERDataset(train_data, tokenizer, neg_pos_ratio=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    confidence, scores = {}, []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Estimating span confidence"):
            all_probs = []
            for voter in voters:
                logits = voter.get_span_logits(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device), batch["span_starts"], batch["span_ends"])
                all_probs.append([None if item is None else torch.softmax(item, -1).cpu()
                                  for item in logits])
            for i, sample_idx in enumerate(batch["idx"]):
                sample_map = {}
                voter_probs = [items[i] for items in all_probs if items[i] is not None]
                for span_id, label_tensor in enumerate(batch["span_labels"][i]):
                    label = int(label_tensor)
                    if label == 0 or not voter_probs:
                        continue
                    gold_prob = float(np.mean([float(item[span_id, label])
                                               for item in voter_probs]))
                    agreement = sum(int(item[span_id].argmax()) == label
                                    for item in voter_probs) / len(voter_probs)
                    score = gold_prob * agreement
                    start = int(batch["span_starts"][i][span_id])
                    end = int(batch["span_ends"][i][span_id])
                    sample_map[f"{start}:{end}"] = score
                    scores.append(score)
                confidence[str(sample_idx)] = sample_map
    thresholds = [float(np.percentile(scores or [0.0], value))
                  for value in args.curriculum_percentiles]
    return {"span_confidence": confidence, "thresholds": thresholds}


def curriculum_weights(batch, confidence, threshold, minimum):
    result = []
    for i, sample_idx in enumerate(batch["idx"]):
        sample_map = confidence["span_confidence"].get(str(sample_idx), {})
        weights = []
        for start, end, label in zip(batch["span_starts"][i], batch["span_ends"][i],
                                     batch["span_labels"][i]):
            if int(label) == 0:
                weights.append(1.0)
                continue
            score = sample_map.get(f"{int(start)}:{int(end)}", minimum)
            weights.append(1.0 if score >= threshold else max(minimum, score))
        result.append(torch.tensor(weights, dtype=torch.float32))
    return result


def train_complete_model(train_data, tokenizer, output_dir, args, device, logger):
    voters = train_voters(train_data, tokenizer, output_dir, args, device, logger)
    confidence = estimate_positive_confidence(train_data, tokenizer, voters, args, device)
    with (output_dir / "positive_confidence.json").open("w", encoding="utf-8") as stream:
        json.dump(confidence, stream, ensure_ascii=False, indent=2)
    set_seed(args.seed)
    model = SpanBERTForNER(MODEL_PATH, NUM_LABELS, MAX_SPAN_LENGTH).to(device)
    dataset = NERDataset(train_data, tokenizer)
    for stage, threshold in enumerate(confidence["thresholds"]):
        epochs = args.stage_epochs[min(stage, len(args.stage_epochs) - 1)]
        decay = args.learning_rate_decay[min(stage, len(args.learning_rate_decay) - 1)]
        generator = torch.Generator().manual_seed(args.seed + stage)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, generator=generator)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate * decay, weight_decay=0.01)
        steps = max(1, len(loader) * epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for batch in tqdm(loader, desc=f"Curriculum stage {stage + 1}", leave=False):
                weights = curriculum_weights(batch, confidence, threshold,
                                             args.minimum_positive_weight)
                loss = weighted_forward(model, batch, device, weights)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total += loss.item()
            logger.info("Stage %d epoch %d loss %.4f", stage + 1, epoch + 1,
                        total / max(len(loader), 1))
    torch.save(model.state_dict(), output_dir / "model.pt")
    return model


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [precision, recall, f1]


def evaluate(model, test_data, tokenizer, args, device):
    loader = DataLoader(NERDataset(test_data, tokenizer, neg_pos_ratio=0),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    predictions = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader, desc="Final test evaluation"):
            predictions.extend(model.predict_spans(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device), batch["offset_mapping"], batch["texts"],
                threshold=args.prediction_threshold))
    counts = defaultdict(int)
    for predicted, sample in zip(predictions, test_data):
        pred = {(x["start"], x["end"], x["type"]) for x in predicted}
        gold = {(x["start"], x["end"], x["entity_type"]) for x in sample["entities"]}
        counts["strict_tp"] += len(pred & gold)
        counts["strict_fp"] += len(pred - gold)
        counts["strict_fn"] += len(gold - pred)
        matched_pred, matched_gold = set(), set()
        for gold_item in gold:
            for pred_item in pred:
                overlap = max(0, min(gold_item[1], pred_item[1]) - max(gold_item[0], pred_item[0]))
                union = max(gold_item[1], pred_item[1]) - min(gold_item[0], pred_item[0])
                if (pred_item[2] == gold_item[2] and pred_item not in matched_pred and union
                        and overlap / union >= 0.5):
                    matched_pred.add(pred_item)
                    matched_gold.add(gold_item)
                    break
        counts["partial_tp"] += len(matched_gold)
        counts["partial_fp"] += len(pred) - len(matched_pred)
        counts["partial_fn"] += len(gold) - len(matched_gold)
    return ({"strict": prf(counts["strict_tp"], counts["strict_fp"], counts["strict_fn"]),
             "partial": prf(counts["partial_tp"], counts["partial_fp"], counts["partial_fn"]),
             "counts": dict(counts)}, predictions)


def parse_list(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train the complete LLMCPNER method.")
    parser.add_argument("--train-file", required=True, help="Weakly supervised training JSON.")
    parser.add_argument("--test-file", default=str(DATA_DIR / "corrected_test_gold.json"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "complete_experiment"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--prediction-threshold", type=float, default=0.3)
    parser.add_argument("--num-voters", type=int, default=3)
    parser.add_argument("--voter-sample-ratio", type=float, default=0.15)
    parser.add_argument("--voter-epochs", type=int, default=2)
    parser.add_argument("--minimum-positive-weight", type=float, default=0.3)
    parser.add_argument("--curriculum-percentiles", type=lambda x: parse_list(x, int),
                        default=[50, 80, 100])
    parser.add_argument("--stage-epochs", type=lambda x: parse_list(x, int), default=[1, 1, 2])
    parser.add_argument("--learning-rate-decay", type=lambda x: parse_list(x, float),
                        default=[1.0, 0.8, 0.5])
    args = parser.parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_data = normalize_samples(read_json(args.train_file))
    test_data = normalize_samples(read_json(args.test_file), keep_empty=True)
    if not train_data or not test_data:
        raise ValueError("Training and test files must contain valid samples.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = train_complete_model(train_data, tokenizer, output_dir, args, device, logger)
    metrics, predictions = evaluate(model, test_data, tokenizer, args, device)
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump({"metrics": metrics, "configuration": vars(args)}, stream,
                  ensure_ascii=False, indent=2)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as stream:
        json.dump(predictions, stream, ensure_ascii=False, indent=2)
    logger.info("Final strict P/R/F1: %s", metrics["strict"])
    logger.info("Final partial P/R/F1: %s", metrics["partial"])


if __name__ == "__main__":
    main()
