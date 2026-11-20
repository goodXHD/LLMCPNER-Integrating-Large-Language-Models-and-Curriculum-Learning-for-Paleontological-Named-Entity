from pathlib import Path
import sys,os,json,time,math,random,importlib.util,shutil
import torch
import argparse
from types import SimpleNamespace
import engine
import runtime
import weighting as old
from runtime import sha
ROOT=Path(__file__).resolve().parent
NAMES=['ordinary','static_confidence','class_cumulative','random_cumulative']
def save(p,d):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def snapshot(p,d):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');torch.save(d,tmp);tmp.replace(p)
def lr_factor(step,per_epoch):
 warm=max(1,math.ceil(.1*per_epoch));end=10*per_epoch
 if step<warm:return (step+1)/warm
 return max(.1,1-.9*(step-warm)/max(1,end-warm))
def stop(epoch,bad):return epoch>=2 and bad>=3

def main(name,dev):
 assert name in NAMES
 plan=json.loads(Path(CLI.config).read_text()); plan['input_sha256']={p:sha(p) for p in [CLI.train_file,CLI.dev_file]+([CLI.memberships] if name!='ordinary' else [])}
 args=SimpleNamespace(**plan,train_file=CLI.train_file,test_file=CLI.dev_file,encoder=CLI.encoder,device=dev,memory_fraction=CLI.memory_fraction);runtime.configure_runtime(args);torch.set_num_threads(2);device=torch.device(dev)
 out=Path(CLI.output_dir);out.mkdir(parents=True,exist_ok=False)
 samples=engine.read_samples(args.train_file);gold=engine.read_json(args.test_file);tokenizer=runtime.AutoTokenizer.from_pretrained(args.encoder,local_files_only=True)
 package={} if name=='ordinary' else engine.read_json(CLI.memberships)
 if name!='ordinary' and package['train_sha256']!=sha(CLI.train_file):raise ValueError('Confidence cache does not match training data')
 membership=package.get('spans',{})
 engine.seed_all(plan['seed']);model=engine.new_model(args,device);data=engine.NERDataset(samples,tokenizer,neg_pos_ratio=args.negative_ratio);per_epoch=math.ceil(len(data)/args.batch_size)
 optim=engine.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01);updates=0;metrics=[];exposures=[];stages=[];started=time.monotonic();total_epochs=0
 save(out/'protocol.json',{'condition':name,**plan,'steps_per_epoch':per_epoch})
 for stage in range(3):
  if stage:
   state=torch.load(out/f'stage_{stage}_best.pt',map_location=device,weights_only=False)
   model.load_state_dict(state['model'],strict=True);optim.load_state_dict(state['optimizer']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device);del state
  # Preserve Adam moments at the prior stage best; explicitly restart stage-local LR schedule.
  for group in optim.param_groups:group['lr']=args.learning_rate;group['initial_lr']=args.learning_rate
  scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda step:lr_factor(step,per_epoch))
  best=-1.;best_epoch=0;bad=0;epoch=0;stage_updates=0
  while True:
   epoch+=1;total_epochs+=1;seed=plan['seed']+stage*100000+epoch-1;engine.seed_all(seed);model.train();loss_sum=0.;active=positive=negative=0;dose={}
   for bi,batch in enumerate(engine.loader(data,args.batch_size,seed),1):
    ws,masks=old.weights_and_mask(batch,stage,name,membership);optim.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     z=model.get_span_logits(batch['input_ids'].to(device),batch['attention_mask'].to(device),batch['token_type_ids'].to(device),batch['span_starts'],batch['span_ends']);loss=old.weighted_loss(z,batch['span_labels'],ws,masks)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();scheduler.step();updates+=1;stage_updates+=1;loss_sum+=float(loss.detach())
    for ys,w,mask in zip(batch['span_labels'],ws,masks):
     active+=int(((ys>0)&mask).sum());positive+=int((ys>0).sum());negative+=int((ys==0).sum())
     for label,weight in zip(ys.tolist(),w.tolist()):
      if label:dose[str(label)]=dose.get(str(label),0.)+weight
    if bi%25==0:save(out/'status.json',dict(status='training',stage=stage+1,epoch=epoch,total_epochs=total_epochs,batch=bi,batches_per_epoch=per_epoch,updates=updates,stage_updates=stage_updates,best_F1=best if best>=0 else None,best_epoch=best_epoch,bad_epochs=bad,learning_rate=optim.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-started))
   save(out/'status.json',dict(status='evaluating',stage=stage+1,epoch=epoch,total_epochs=total_epochs,updates=updates,best_F1=best if best>=0 else None,bad_epochs=bad))
   target=out/f'stage_{stage+1}'/f'epoch_{epoch:03d}';target.mkdir(parents=True)
   m=engine.final_evaluation(model,tokenizer,gold,args,device,target);score=100*m['strict']['f1'];improved=score>best
   if improved:best=score;best_epoch=epoch;bad=0
   else:bad+=1
   row=dict(stage=stage+1,epoch=epoch,total_epochs=total_epochs,updates=updates,strict_P=100*m['strict']['precision'],strict_R=100*m['strict']['recall'],strict_F1=score,partial_F1=100*m['partial']['f1'],improved=improved,best_epoch=best_epoch,bad_epochs=bad,learning_rate=optim.param_groups[0]['lr'])
   metrics.append(row);save(out/'epoch_metrics.json',metrics)
   exposures.append(dict(stage=stage+1,epoch=epoch,positive_admitted=active,positive_presented=positive,negative_always_active=negative,positive_weight_sums_by_type=dose,loss_sum=loss_sum));save(out/'exposure_history.json',exposures)
   if improved:
    snapshot(out/f'stage_{stage+1}_best.pt',dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),stage=stage+1,epoch=epoch,updates_at_checkpoint=updates,metrics=row))
    save(out/f'stage_{stage+1}_best.json',row)
   # A last-epoch full state supports recovery; cumulative actual updates never roll back.
   snapshot(out/'resume.pt',dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),stage=stage+1,epoch=epoch,updates=updates,total_epochs=total_epochs,best_F1=best,best_epoch=best_epoch,bad_epochs=bad))
   save(out/'status.json',dict(status='epoch_complete',**row,best_F1=best,elapsed_seconds=time.monotonic()-started))
   if stop(epoch,bad):break
  stages.append(dict(stage=stage+1,epochs=epoch,best_epoch=best_epoch,best_F1=best,actual_stage_updates=stage_updates,stop='3 consecutive non-improving development evaluations, minimum2 epochs'))
  save(out/'stage_summary.json',stages)
 # Final endpoint is stage3 best, not maximum across stages.
 state=torch.load(out/'stage_3_best.pt',map_location='cpu',weights_only=False);snapshot(out/'model.pt',state['model']);chosen=out/'stage_3'/f'epoch_{state["epoch"]:03d}'
 for f in ['metrics.json','predictions.json']:shutil.copy2(chosen/f,out/f)
 save(out/'verification.json',dict(complete=True,actual_updates=updates,total_epochs=total_epochs,selected_stage=3,selected_epoch=state['epoch'],train_sha256=sha(args.train_file),reference_sha256=sha(args.test_file),selection='stage3 best development strict F1; reused development reference',stage_summaries=stages))
 save(out/'status.json',dict(status='complete',stage=3,epoch=epoch,total_epochs=total_epochs,updates=updates,last_metrics=state['metrics'],elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description='Three-stage confidence-weighted span training')
 parser.add_argument('--config',default='configs/full_method.json')
 parser.add_argument('--train-file',required=True)
 parser.add_argument('--dev-file',required=True,help='Labeled development data used for early stopping; use a separate set for final evaluation.')
 parser.add_argument('--memberships',default='outputs/confidence/memberships.json')
 parser.add_argument('--encoder',required=True,help='Pinned local SciBERT directory')
 parser.add_argument('--output-dir',required=True)
 parser.add_argument('--condition',choices=NAMES,default='class_cumulative')
 parser.add_argument('--device',default='cuda:0')
 parser.add_argument('--memory-fraction',type=float,default=.9)
 CLI=parser.parse_args()
 if Path(CLI.output_dir).exists():parser.error('Output directory already exists; choose a new directory.')
 try:main(CLI.condition,CLI.device)
 except Exception as exc:
  out=Path(CLI.output_dir);out.mkdir(exist_ok=True,parents=True)
  save(out/'status.json',{'status':'failed','error':repr(exc)});raise
