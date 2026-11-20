"""
数据集模块
包含 NERDataset 和 collate_fn
"""

import random
import torch
from torch.utils.data import Dataset

from config import MAX_LENGTH, MAX_SPAN_LENGTH, NEG_POS_RATIO, LABEL2ID, TYPE_MAPPING


class NERDataset(Dataset):
    """NER数据集"""

    def __init__(self, data, tokenizer, max_length=MAX_LENGTH, max_span_length=MAX_SPAN_LENGTH, neg_pos_ratio=NEG_POS_RATIO):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_span_length = max_span_length
        self.neg_pos_ratio = neg_pos_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item.get('entities', [])

        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length',
                                  truncation=True, return_offsets_mapping=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).squeeze()
        offset_mapping = encoding['offset_mapping'].squeeze().tolist()

        # 字符到token映射
        char_to_token = {}
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char == 0 and end_char == 0:
                continue
            for char_idx in range(start_char, end_char):
                char_to_token[char_idx] = token_idx

        # 实体跨度映射
        entity_spans = {}
        for entity in entities:
            start_char = entity.get('start', -1)
            end_char = entity.get('end', -1)
            entity_type = TYPE_MAPPING.get(entity.get('entity_type', entity.get('type', '')), '')

            if start_char < 0 or end_char < 0:
                continue

            start_token = char_to_token.get(start_char)
            end_token = char_to_token.get(end_char - 1)

            if start_token is not None and end_token is not None:
                label = LABEL2ID.get(entity_type, 0)
                if label > 0:
                    entity_spans[(start_token, end_token)] = label

        # 生成正负样本
        seq_len = (attention_mask == 1).sum().item()
        positive_spans = []
        negative_spans = []

        for start in range(1, seq_len - 1):
            for end in range(start, min(start + self.max_span_length, seq_len - 1)):
                if (start, end) in entity_spans:
                    positive_spans.append((start, end, entity_spans[(start, end)]))
                else:
                    negative_spans.append((start, end, 0))

        # 采样负样本
        if self.neg_pos_ratio == 0:
            # 评估时不采负样本
            num_neg = 0
        else:
            num_neg = min(len(negative_spans), max(len(positive_spans) * self.neg_pos_ratio, 50))
        sampled_neg = random.sample(negative_spans, num_neg) if negative_spans and num_neg < len(negative_spans) else negative_spans
        selected_spans = positive_spans + sampled_neg
        random.shuffle(selected_spans)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'span_starts': torch.tensor([s[0] for s in selected_spans], dtype=torch.long),
            'span_ends': torch.tensor([s[1] for s in selected_spans], dtype=torch.long),
            'span_labels': torch.tensor([s[2] for s in selected_spans], dtype=torch.long),
            'offset_mapping': offset_mapping,
            'text': text,
            'entities': entities,
            'idx': idx
        }


def collate_fn(batch):
    """批次整理函数"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
        'span_starts': [item['span_starts'] for item in batch],
        'span_ends': [item['span_ends'] for item in batch],
        'span_labels': [item['span_labels'] for item in batch],
        'offset_mapping': [item['offset_mapping'] for item in batch],
        'texts': [item['text'] for item in batch],
        'entities': [item['entities'] for item in batch],
        'idx': [item['idx'] for item in batch]
    }
