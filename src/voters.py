"""Train one of the final three 60%-bootstrap voters and score training spans."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,math,time,random,torch
import torch.nn.functional as F
import engine,runtime

def save(path,value):engine.save_json(path,value)
def span_key(a,b,y):return f'{int(a)}:{int(b)}:{int(y)}'

def included_loss(logits,labels,weights):
 num=None;den=0
 for z,y,w in zip(logits,labels,weights):
  if z is None:continue
  y=y.to(z.device);w=w.to(z.device);v=(F.cross_entropy(z,y,reduction='none')*w).sum();num=v if num is None else num+v;den+=int((w>0).sum())
 if num is None or den==0:raise RuntimeError('No active spans')
 return num/den

def main():
 args=SimpleNamespace(**engine.read_json(CLI.config),encoder=CLI.encoder,train_file=CLI.train_file,device=CLI.device,memory_fraction=CLI.memory_fraction)
 device=torch.device(CLI.device);runtime.configure_runtime(args)
 tokenizer=runtime.AutoTokenizer.from_pretrained(args.encoder,local_files_only=True)
 started=time.monotonic();name=CLI.member;plan={'voter_data':CLI.original_file}
 member=int(name);out=Path(CLI.output_dir);out.mkdir(parents=True,exist_ok=False);data=engine.read_samples(plan['voter_data']);rng=random.Random(CLI.seed+7919*member)
 pools={label:[] for label in ['taxa','location','section','strata','lithology','facies','age']}
 for i,sample in enumerate(data):
  for label in {e.get('entity_type',e.get('type')) for e in sample['entities']}:
   if label in pools:pools[label].append(i)
 assert all(pools.values())
 indices=[];target=round(.6*len(data))
 for draw in range(target):
  if draw < target//2:indices.append(rng.randrange(len(data)))
  else:indices.append(rng.choice(pools[rng.choice(list(pools))]))
 rng.shuffle(indices);seed=CLI.seed+1009*member
 save(out/'bootstrap_indices.json',{'seed':seed,'draws_with_replacement':indices,'population':len(data),'fraction':.6,'sampling':'half uniform draws, half uniformly select type then containing sentence; replacement; shuffled','type_pool_sizes':{k:len(v) for k,v in pools.items()}})
 selected=[data[i] for i in indices];engine.seed_all(seed);model=engine.new_model(args,device);ds=engine.NERDataset(selected,tokenizer,neg_pos_ratio=args.negative_ratio);optim=engine.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01);steps=math.ceil(len(ds)/args.batch_size)*2;scheduler=engine.get_linear_schedule_with_warmup(optim,int(.1*steps),steps);updates=0
 for ep in range(2):
  engine.seed_all(seed+ep);model.train()
  for batch in engine.loader(ds,args.batch_size,seed+ep):
   optim.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    z=model.get_span_logits(batch['input_ids'].to(device),batch['attention_mask'].to(device),batch['token_type_ids'].to(device),batch['span_starts'],batch['span_ends']);loss=included_loss(z,batch['span_labels'],[torch.ones_like(y,dtype=torch.float32) for y in batch['span_labels']])
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();scheduler.step();updates+=1
   if updates%25==0:save(out/'status.json',{'status':'training_voter','epoch':ep+1,'updates':updates,'total_updates':steps,'elapsed_seconds':time.monotonic()-started})
 torch.save(model.state_dict(),out/'model.pt');model.eval();samples=engine.read_samples(args.train_file);positive=engine.PositiveDataset(samples,tokenizer,neg_pos_ratio=0);result={};done=0
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for batch in engine.loader(positive,args.batch_size,0,False):
   z=model.get_span_logits(batch['input_ids'].to(device),batch['attention_mask'].to(device),batch['token_type_ids'].to(device),batch['span_starts'],batch['span_ends'])
   for j,idx in enumerate(batch['idx']):
    ys=batch['span_labels'][j];keys=[span_key(a,b,y) for a,b,y in zip(batch['span_starts'][j],batch['span_ends'][j],ys)];order=sorted(range(len(keys)),key=lambda j:keys[j]);probs=z[j].float().softmax(-1).cpu() if len(ys) else torch.empty(0,8)
    result[str(idx)]={'keys':[keys[k] for k in order],'probabilities':probs[order]}
   done+=len(batch['idx'])
   if done%100==0:save(out/'status.json',{'status':'scoring','texts':done,'total_texts':len(samples)})
 save(out/'provenance.json',{'train_sha256':runtime.sha(CLI.train_file),'original_sha256':runtime.sha(CLI.original_file),'member':member,'model_sha256':runtime.sha(out/'model.pt')});torch.save(result,out/'probabilities.pt');save(out/'status.json',{'status':'complete','voter_epochs':2,'optimizer_updates':updates,'scored_texts':len(samples),'elapsed_seconds':time.monotonic()-started});return

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--member',type=int,choices=[0,1,2],required=True)
 p.add_argument('--original-file',required=True)
 p.add_argument('--train-file',required=True)
 p.add_argument('--encoder',required=True)
 p.add_argument('--config',default='configs/full_method.json')
 p.add_argument('--output-dir',required=True)
 p.add_argument('--device',default='cuda:0')
 p.add_argument('--memory-fraction',type=float,default=.9)
 CLI=p.parse_args();CLI.seed=engine.read_json(CLI.config)['seed']
 if Path(CLI.output_dir).exists():p.error('Output directory already exists; choose a new directory.')
 main()
