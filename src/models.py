"""
模型定义模块
包含 SpanEmbedding 和 SpanBERTForNER
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class SpanEmbedding(nn.Module):
    """跨度嵌入: 起始token + 结束token + 宽度嵌入"""

    def __init__(self, hidden_size, max_span_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_span_length = max_span_length
        self.width_embedding = nn.Embedding(max_span_length + 1, hidden_size)
        nn.init.normal_(self.width_embedding.weight, mean=0.0, std=0.02)

    def forward(self, sequence_output, batch_idx, span_starts, span_ends):
        start_emb = sequence_output[batch_idx, span_starts]
        end_emb = sequence_output[batch_idx, span_ends]
        span_widths = torch.clamp(span_ends - span_starts + 1, 0, self.max_span_length)
        width_emb = self.width_embedding(span_widths)
        return torch.cat([start_emb, end_emb, width_emb], dim=-1)

class SpanBERTForNER(nn.Module):
    """基于跨度的BERT NER模型"""

    def __init__(self, model_path, num_labels, max_span_length=12, load_encoder_weights=True):
        super().__init__()
        self.num_labels = num_labels
        self.max_span_length = max_span_length
        self.config = AutoConfig.from_pretrained(model_path)
        if load_encoder_weights:
            self.bert = AutoModel.from_pretrained(model_path, config=self.config, attn_implementation='eager')
        else:
            self.config._attn_implementation = 'eager'
            self.bert = AutoModel.from_config(self.config)
        self.hidden_size = self.config.hidden_size
        self.span_embedding = SpanEmbedding(self.hidden_size, max_span_length)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.hidden_size, num_labels))
        self.dropout = nn.Dropout(0.1)

    def predict_spans(self, input_ids, attention_mask, token_type_ids=None, offset_mapping=None, texts=None, threshold=0.5):
        """预测实体跨度"""
        self.eval()
        batch_size = input_ids.size(0)
        all_predictions = []
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = outputs.last_hidden_state
            for i in range(batch_size):
                seq_len = (attention_mask[i] == 1).sum().item()
                text = texts[i] if texts else ''
                mapping = offset_mapping[i] if offset_mapping else None
                span_starts_list, span_ends_list = ([], [])
                for start in range(1, seq_len - 1):
                    for end in range(start, min(start + self.max_span_length, seq_len - 1)):
                        span_starts_list.append(start)
                        span_ends_list.append(end)
                if not span_starts_list:
                    all_predictions.append([])
                    continue
                span_starts_t = torch.tensor(span_starts_list, device=input_ids.device)
                span_ends_t = torch.tensor(span_ends_list, device=input_ids.device)
                span_emb = self.span_embedding(sequence_output, i, span_starts_t, span_ends_t)
                logits = self.classifier(span_emb)
                probs = torch.softmax(logits, dim=-1)
                pred_labels = probs.argmax(dim=-1)
                pred_scores = probs.max(dim=-1).values
                predictions = []
                for start, end, label, score in zip(span_starts_list, span_ends_list, pred_labels.tolist(), pred_scores.tolist()):
                    if label > 0 and score > threshold:
                        if mapping and start < len(mapping) and (end < len(mapping)):
                            start_char, end_char = (mapping[start][0], mapping[end][1])
                            if start_char is not None and end_char is not None:
                                predictions.append({'start': start_char, 'end': end_char, 'text': text[start_char:end_char], 'type': ID2LABEL[label], 'score': score})
                predictions.sort(key=lambda x: x['score'], reverse=True)
                predictions = self._remove_overlapping(predictions)
                all_predictions.append(predictions)
        return all_predictions

    def _remove_overlapping(self, predictions):
        if not predictions:
            return predictions
        filtered = []
        for pred in sorted(predictions, key=lambda x: x['score'], reverse=True):
            if not any((not (pred['end'] <= existing['start'] or pred['start'] >= existing['end']) for existing in filtered)):
                filtered.append(pred)
        return filtered

    def get_span_logits(self, input_ids, attention_mask, token_type_ids, all_span_starts, all_span_ends):
        """Respect caller mode/autograd: training and MC scoring share this path.

        Callers own eval/no_grad. The legacy forced eval/no_grad here silently
        disabled MC dropout and prevented this method from supporting training.
        """
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        sequence_output = self.dropout(sequence_output)
        all_logits = []
        for i, (starts, ends) in enumerate(zip(all_span_starts, all_span_ends)):
            if len(starts) == 0:
                all_logits.append(None)
                continue
            span_emb = self.span_embedding(sequence_output, i, starts.to(input_ids.device), ends.to(input_ids.device))
            all_logits.append(self.classifier(span_emb))
        return all_logits
from config import ID2LABEL
