"""Common model/data helpers extracted from the final experiment's engine."""
import json, random
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from config import ENTITY_TYPES, NUM_LABELS, MAX_SPAN_LENGTH
from dataset import NERDataset, collate_fn
from models import SpanBERTForNER
from metrics import evaluate_predictions
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def read_samples(path):
    samples = read_json(path)
    if not isinstance(samples, list) or not samples:
        raise ValueError('Expected a nonempty list of samples')
    for i, sample in enumerate(samples):
        text = sample.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty source text at sample {i}')
        entities = sample.get('entities')
        if not isinstance(entities, list):
            raise ValueError(f'Missing entities at sample {i}')
        seen = {}
        for entity in entities:
            label = entity.get('entity_type', entity.get('type'))
            a, b = entity['start'], entity['end']
            if (label not in ENTITY_TYPES or type(a) is not int or type(b) is not int
                    or not 0 <= a < b <= len(text)):
                raise ValueError(f'Invalid label/offset at sample {i}')
            if 'text' in entity and entity['text'] != text[a:b]:
                raise ValueError(f'Offset/text mismatch at sample {i}')
            if (a, b) in seen:
                raise ValueError(f'Duplicate/conflicting span at sample {i}')
            seen[a, b] = label
            entity['entity_type'] = label
    return samples

class PositiveDataset(NERDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        keep = item['span_labels'] > 0
        for key in ('span_starts', 'span_ends', 'span_labels'):
            item[key] = item[key][keep]
        return item

def loader(dataset, batch_size, seed, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))

def new_model(args, device):
    return SpanBERTForNER(args.encoder, NUM_LABELS, MAX_SPAN_LENGTH).to(device)

def final_evaluation(model, tokenizer, samples, args, device, out):
    dataset = NERDataset(samples, tokenizer, neg_pos_ratio=0)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader(dataset, args.batch_size, 0, False):
            predictions.extend(model.predict_spans(batch['input_ids'].to(device),
                batch['attention_mask'].to(device), batch['token_type_ids'].to(device),
                batch['offset_mapping'], batch['texts'], args.prediction_threshold))
    metrics = evaluate_predictions(samples, predictions)
    save_json(out / 'predictions.json', [{'text': s['text'], 'pred_entities': p}
                                         for s, p in zip(samples, predictions)])
    save_json(out / 'metrics.json', metrics)
    return metrics
