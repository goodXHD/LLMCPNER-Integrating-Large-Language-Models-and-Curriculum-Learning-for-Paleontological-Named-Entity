"""The weighting and normalization used by the final method."""
import torch
import torch.nn.functional as F

def weights_and_mask(batch,stage,condition,membership):
 ws=[];ms=[]
 for idx,starts,ends,ys in zip(batch['idx'],batch['span_starts'],batch['span_ends'],batch['span_labels']):
  weights=[];active=[]
  for a,b,y in zip(starts.tolist(),ends.tolist(),ys.tolist()):
   if y==0 or condition=='ordinary':weights.append(1.);active.append(True);continue
   value=membership[str(idx)][f'{a}:{b}:{y}'];key={'global_cumulative':'global_stage','class_cumulative':'class_stage','random_cumulative':'random_stage'}.get(condition)
   included=True if condition=='static_confidence' else value[key]<=stage
   weights.append(value['confidence'] if included else 0.);active.append(included)
  ws.append(torch.tensor(weights,dtype=torch.float32));ms.append(torch.tensor(active,dtype=torch.bool))
 return ws,ms

def weighted_loss(logits,labels,weights,masks):
 numerator=None;denominator=0
 for z,y,w,mask in zip(logits,labels,weights,masks):
  if z is None:continue
  y=y.to(z.device);w=w.to(z.device);v=(F.cross_entropy(z,y,reduction='none')*w).sum();numerator=v if numerator is None else numerator+v;denominator+=int(mask.sum())
 if numerator is None or not denominator:raise RuntimeError('Empty active span batch')
 return numerator/denominator
