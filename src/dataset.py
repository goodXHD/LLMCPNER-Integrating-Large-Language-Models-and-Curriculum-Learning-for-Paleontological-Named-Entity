"""Dataset utilities for character-offset, span-based NER."""

import random

import torch
from torch.utils.data import Dataset

from config import LABEL2ID, MAX_LENGTH, MAX_SPAN_LENGTH, NEG_POS_RATIO, TYPE_MAPPING


class NERDataset(Dataset):
    def __init__(self, samples, tokenizer, neg_pos_ratio=NEG_POS_RATIO):
        self.samples = samples
        self.tokenizer = tokenizer
        self.neg_pos_ratio = neg_pos_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        encoded = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").squeeze(0)
        gold = {}
        for entity in sample.get("entities", []):
            label = entity.get("entity_type", entity.get("type", entity.get("label", "")))
            label = TYPE_MAPPING.get(label, label)
            if label in LABEL2ID:
                gold[(int(entity["start"]), int(entity["end"]))] = LABEL2ID[label]

        valid = [i for i, (start, end) in enumerate(offsets.tolist()) if end > start]
        positives, negatives = [], []
        for left_pos, start_token in enumerate(valid):
            for end_token in valid[left_pos:left_pos + MAX_SPAN_LENGTH]:
                char_start = int(offsets[start_token, 0])
                char_end = int(offsets[end_token, 1])
                item = (start_token, end_token, gold.get((char_start, char_end), 0))
                (positives if item[2] else negatives).append(item)

        if self.neg_pos_ratio and positives:
            limit = min(len(negatives), len(positives) * self.neg_pos_ratio)
            negatives = random.sample(negatives, limit)
        spans = positives + negatives
        spans.sort(key=lambda value: (value[0], value[1]))
        return {
            "idx": idx,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).squeeze(0),
            "offset_mapping": offsets,
            "span_starts": torch.tensor([x[0] for x in spans], dtype=torch.long),
            "span_ends": torch.tensor([x[1] for x in spans], dtype=torch.long),
            "span_labels": torch.tensor([x[2] for x in spans], dtype=torch.long),
            "text": text,
            "entities": sample.get("entities", []),
        }


def collate_fn(batch):
    return {
        "idx": [item["idx"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "token_type_ids": torch.stack([item["token_type_ids"] for item in batch]),
        "offset_mapping": [item["offset_mapping"] for item in batch],
        "span_starts": [item["span_starts"] for item in batch],
        "span_ends": [item["span_ends"] for item in batch],
        "span_labels": [item["span_labels"] for item in batch],
        "texts": [item["text"] for item in batch],
        "entities": [item["entities"] for item in batch],
    }

