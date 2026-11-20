from pathlib import Path
import json,torch,random,collections
import argparse
p=argparse.ArgumentParser(description='Aggregate three voter probabilities into confidence and class curricula')
p.add_argument('--voter-dirs',nargs=3,required=True)
p.add_argument('--output-dir',required=True)
p.add_argument('--seed',type=int,default=13)
args=p.parse_args()
ROOT=Path(args.output_dir)
if ROOT.exists():p.error('Output directory already exists; choose a new directory.')
provenance=[json.loads((Path(d)/'provenance.json').read_text()) for d in args.voter_dirs]
if len({v['train_sha256'] for v in provenance})!=1:raise ValueError('Voters scored different training data')
if sorted(v['member'] for v in provenance)!=[0,1,2]:raise ValueError('Expected three separately trained voter members 0,1,2')
ROOT.mkdir(parents=True)
ps=[torch.load(Path(d)/'probabilities.pt',map_location='cpu',weights_only=True) for d in args.voter_dirs]
assert all(set(x)==set(ps[0]) for x in ps);spans={};flat=[]
for idx in ps[0]:
 rows=[x[idx] for x in ps];assert all(x['keys']==rows[0]['keys'] for x in rows);keys=rows[0]['keys'];spans[idx]={}
 if not keys:continue
 p=torch.stack([x['probabilities'] for x in rows]);labels=torch.tensor([int(k.rsplit(':',1)[1]) for k in keys]);c=(p.mean(0).gather(1,labels[:,None]).squeeze(1)+(p.argmax(-1)==labels[None,:]).float().mean(0))/2
 for key,v in zip(keys,c):spans[idx][key]={'confidence':float(v)};flat.append((idx,key,float(v),int(key.rsplit(':',1)[1])))
values=torch.tensor([x[2] for x in flat],dtype=torch.float64);globalq=[float(torch.quantile(values,.7)),float(torch.quantile(values,.3))];classq={};audit=[]
for label in range(1,8):
 rows=[x for x in flat if x[3]==label];v=torch.tensor([x[2] for x in rows],dtype=torch.float64);q=[float(torch.quantile(v,.7)),float(torch.quantile(v,.3))];classq[str(label)]=q
 for idx,key,c,_ in rows:
  spans[idx][key]['global_stage']=0 if c>=globalq[0] else 1 if c>=globalq[1] else 2
  spans[idx][key]['class_stage']=0 if c>=q[0] else 1 if c>=q[1] else 2
 shuffled=rows.copy();random.Random(args.seed+label*100003).shuffle(shuffled);stages=sorted(spans[i][k]['class_stage'] for i,k,_,_ in rows)
 for (idx,key,_,_),stage in zip(shuffled,stages):spans[idx][key]['random_stage']=stage
 for stage in range(3):
  counts={field:sum(spans[i][k][field]<=stage for i,k,_,_ in rows) for field in ['global_stage','class_stage','random_stage']};assert counts['class_stage']==counts['random_stage']
  doses={field:sum(c for i,k,c,_ in rows if spans[i][k][field]<=stage) for field in ['global_stage','class_stage','random_stage']}
  audit.append({'label':label,'stage':stage+1,'class_total':len(rows),'counts':counts,'confidence_weight_sums':doses})
(ROOT/'memberships.json').write_text(json.dumps({'train_sha256':provenance[0]['train_sha256'],'score':'(mean probability+agreement)/2','global_quantiles':globalq,'class_quantiles':classq,'spans':spans},ensure_ascii=False))
(ROOT/'inclusion_audit.json').write_text(json.dumps({'global_quantiles':globalq,'class_quantiles':classq,'positive_spans':len(flat),'rows':audit,'random_matching':'same per-type admission counts as class_cumulative, but effective weight sums not forced equal; actual doses reported','ties':'inclusive quantile thresholds; actual admission counts recorded'},ensure_ascii=False,indent=2));print('aggregated',len(flat),'spans',flush=True)
