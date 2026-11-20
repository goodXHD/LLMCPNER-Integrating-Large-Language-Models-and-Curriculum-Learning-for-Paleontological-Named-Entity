"""Validate and merge the separately constructed original and augmented data."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from engine import read_samples
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--original',required=True);p.add_argument('--augmented',required=True);p.add_argument('--output',required=True)
a=p.parse_args();data=read_samples(a.original)+read_samples(a.augmented)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(data,ensure_ascii=False,indent=2))
print(json.dumps({'texts':len(data),'annotations':sum(len(s['entities']) for s in data)}))
