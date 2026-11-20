"""Convert cluster-sampling output to the annotation script's flat input list."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
a=p.parse_args();data=json.loads(Path(a.input).read_text());records=[]
for c in data['clusters']:
 for r in c['samples']:records.append({**r,'cluster_id':c['cluster_id']})
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(records,ensure_ascii=False,indent=2))
