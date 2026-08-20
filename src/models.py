"""SciBERT span classifier used by the paper experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import ID2LABEL


class SpanBERTForNER(nn.Module):
    def __init__(self, model_name, num_labels, max_span_length):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.width_embedding = nn.Embedding(max_span_length + 1, 32)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden * 2 + 32, num_labels)
        self.max_span_length = max_span_length

    def span_embedding(self, sequence, batch_index, starts, ends):
        start_repr = sequence[batch_index, starts]
        end_repr = sequence[batch_index, ends]
        widths = (ends - starts + 1).clamp(max=self.max_span_length)
        return torch.cat((start_repr, end_repr, self.width_embedding(widths)), dim=-1)

    def get_span_logits(self, input_ids, attention_mask, token_type_ids, all_starts, all_ends):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence = self.dropout(output.last_hidden_state)
        logits = []
        for i, (starts, ends) in enumerate(zip(all_starts, all_ends)):
            if starts.numel() == 0:
                logits.append(None)
                continue
            embedding = self.span_embedding(sequence, i, starts.to(sequence.device), ends.to(sequence.device))
            logits.append(self.classifier(embedding))
        return logits

    def forward_with_probs(self, input_ids, attention_mask, token_type_ids, all_span_starts,
                           all_span_ends, all_span_labels, teacher_probs=None,
                           temperature=1.0, alpha=1.0):
        logits = self.get_span_logits(input_ids, attention_mask, token_type_ids,
                                      all_span_starts, all_span_ends)
        losses = []
        for item_logits, labels in zip(logits, all_span_labels):
            if item_logits is not None:
                losses.append(F.cross_entropy(item_logits, labels.to(item_logits.device)))
        loss = torch.stack(losses).mean() if losses else input_ids.sum() * 0.0
        return {"loss": loss, "logits": logits}

    def predict_spans(self, input_ids, attention_mask, token_type_ids, offset_mapping, texts, threshold=0.3):
        starts, ends = [], []
        for offsets in offset_mapping:
            valid = [i for i, (a, b) in enumerate(offsets.tolist()) if b > a]
            pairs = [(a, b) for pos, a in enumerate(valid) for b in valid[pos:pos + self.max_span_length]]
            starts.append(torch.tensor([p[0] for p in pairs], dtype=torch.long))
            ends.append(torch.tensor([p[1] for p in pairs], dtype=torch.long))
        logits = self.get_span_logits(input_ids, attention_mask, token_type_ids, starts, ends)
        predictions = []
        for sample_logits, sample_starts, sample_ends, offsets, text in zip(logits, starts, ends, offset_mapping, texts):
            entities = []
            if sample_logits is not None:
                probs = torch.softmax(sample_logits, dim=-1)
                scores, labels = probs[:, 1:].max(dim=-1)
                labels = labels + 1
                for score, label, start, end in zip(scores, labels, sample_starts, sample_ends):
                    if float(score) < threshold:
                        continue
                    char_start = int(offsets[start, 0])
                    char_end = int(offsets[end, 1])
                    entities.append({"start": char_start, "end": char_end,
                                     "text": text[char_start:char_end], "type": ID2LABEL[int(label)],
                                     "score": float(score)})
            predictions.append(entities)
        return predictions

