"""Generate separate, fixed-size pools using the recovered three augmentation strategies."""
import argparse,json,os,random,time
from pathlib import Path
import augmentation_strategies as legacy

def validate(row):
 if not row or not row.get('text') or not row.get('entities'):return False
 seen=set()
 for e in row['entities']:
  a,b=e['start'],e['end']
  if not 0<=a<b<=len(row['text']) or e.get('text')!=row['text'][a:b]:return False
  if e['entity_type'] not in legacy.ENTITY_TYPE_DESCRIPTIONS or (a,b) in seen:return False
  seen.add((a,b))
 return True

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',required=True)
 p.add_argument('--entity-pool',required=True,help='JSON object mapping internal type names to lists of entity names')
 p.add_argument('--output',required=True)
 p.add_argument('--replacement',type=int,default=3334)
 p.add_argument('--rewriting',type=int,default=3333)
 p.add_argument('--combination',type=int,default=3333)
 p.add_argument('--model',default=os.environ.get('LLM_MODEL','gpt-5.4-mini'))
 p.add_argument('--seed',type=int,default=13)
 p.add_argument('--attempts-per-sample',type=int,default=10)
 a=p.parse_args()
 if min(a.replacement,a.rewriting,a.combination)<0 or a.attempts_per_sample<1:p.error('Invalid generation budget')
 out=Path(a.output)
 if out.exists():p.error('Output already exists; choose a new path')
 if (a.rewriting or a.combination) and not os.environ.get('OPENAI_API_KEY'):p.error('Set OPENAI_API_KEY for API-backed generation')
 source=json.loads(Path(a.input).read_text());pool=json.loads(Path(a.entity_pool).read_text())
 kw={'api_key':os.environ.get('OPENAI_API_KEY',''),'api_base':os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1'),'model':a.model}
 random.seed(a.seed);out.parent.mkdir(parents=True,exist_ok=True);rows=[]
 jobs=[('entity_substitution',a.replacement,legacy.strategy_entity_substitution),('paraphrase',a.rewriting,legacy.strategy_paraphrase),('entity_combination',a.combination,legacy.strategy_entity_combination)]
 partial=out.with_suffix('.partial.json')
 for name,target,fn in jobs:
  done=attempts=0
  while done<target:
   if attempts>=max(1,target)*a.attempts_per_sample or legacy.should_stop():raise RuntimeError(f'Generation stopped for {name}; accepted data are in {partial}')
   attempts+=1;row=fn(source,pool,kw)
   if validate(row):
    row['id']=f'{name}_{done:06d}';rows.append(row);done+=1
    tmp=partial.with_suffix('.tmp');tmp.write_text(json.dumps(rows,ensure_ascii=False));tmp.replace(partial)
    print(f'{name}: {done}/{target}',flush=True)
 out.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 out.with_suffix('.protocol.json').write_text(json.dumps({'model':a.model,'seed':a.seed,'counts':{n:t for n,t,_ in jobs}},indent=2))
 partial.unlink(missing_ok=True)
if __name__=='__main__':main()
