"""Deterministic character-span evaluation; no test-based model selection."""
from collections import Counter


def evaluate_predictions(gold, predictions):
    if len(gold) != len(predictions):
        raise ValueError('Prediction count differs from test count')
    counts = Counter()
    for sample, predicted in zip(gold, predictions):
        def convert(entities):
            spans = set()
            for e in entities:
                a, b = e['start'], e['end']
                t = e.get('entity_type', e.get('type'))
                if not 0 <= a < b <= len(sample['text']):
                    raise ValueError('Invalid predicted/reference offsets')
                spans.add((t, a, b))
            return sorted(spans)
        g, p = convert(sample['entities']), convert(predicted)
        edges = [[j for j, (u, a, b) in enumerate(p) if t == u and
                  max(0, min(z, b) - max(s, a)) / (max(z, b) - min(s, a)) >= .5]
                 for t, s, z in g]
        matched = {}
        def augment(i, seen):
            for j in edges[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in matched or augment(matched[j], seen):
                    matched[j] = i
                    return True
            return False
        partial = sum(augment(i, set()) for i in range(len(g)))
        counts.update(strict=len(set(g) & set(p)), partial=partial,
                      gold=len(g), predicted=len(p))
    def prf(key):
        tp, ng, np = counts[key], counts['gold'], counts['predicted']
        precision, recall = tp / np if np else 0., tp / ng if ng else 0.
        return {'precision': precision, 'recall': recall,
                'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                'tp': tp, 'fp': np - tp, 'fn': ng - tp}
    return {key: prf(key) for key in ('strict', 'partial')}
