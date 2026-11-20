"""Evaluate a saved checkpoint, or run extraction on unlabeled JSON texts."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
import engine

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True)
 p.add_argument('--encoder',required=True)
 p.add_argument('--input',required=True)
 p.add_argument('--output-dir',required=True)
 p.add_argument('--device',default='cpu')
 p.add_argument('--prediction-threshold',type=float,default=.3)
 p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--unlabeled',action='store_true')
 a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 data=engine.read_json(a.input)
 if a.unlabeled:
  if not data or any(not isinstance(s.get('text'),str) or not s['text'].strip() for s in data):raise ValueError('Expected nonempty JSON records with text')
  data=[{'text':s['text'],'entities':[]} for s in data]
 else:data=engine.read_samples(a.input)
 device=torch.device(a.device)
 # The full state dict already contains encoder weights.
 a.initialize_from_config=True
 model=engine.new_model(a,device)
 model.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=True),strict=True)
 tok=AutoTokenizer.from_pretrained(a.encoder,local_files_only=True)
 result=engine.final_evaluation(model,tok,data,a,device,out)
 if a.unlabeled:(out/'metrics.json').unlink()
 else:print(json.dumps(result,indent=2))
if __name__=='__main__':main()
