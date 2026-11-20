"""Offline check of gradients, annotation matching, and public entry-point imports."""
import sys,tempfile,json,subprocess
from pathlib import Path
import torch
from transformers import BertConfig,BertModel,BertTokenizerFast
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from models import SpanBERTForNER
from weighting import weighted_loss
from metrics import evaluate_predictions
from config import ENTITY_TYPES

def prepare(path):
 enc=path/'encoder';enc.mkdir()
 vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','olenellus','occurs','in','the','formation','.']
 (enc/'vocab.txt').write_text('\n'.join(vocab)+'\n')
 tok=BertTokenizerFast(vocab_file=str(enc/'vocab.txt'),do_lower_case=True);tok.save_pretrained(enc)
 BertModel(BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=512)).save_pretrained(enc)
 samples=[]
 for label in ENTITY_TYPES:
  for n in range(2):samples.append({'text':'Olenellus occurs.','entities':[{'start':0,'end':9,'text':'Olenellus','entity_type':label}]})
 for name in ['original','train','dev']:(path/(name+'.json')).write_text(json.dumps(samples))
 return enc

def main():
 with tempfile.TemporaryDirectory() as temp:
  p=Path(temp);enc=prepare(p)
  model=SpanBERTForNER(str(enc),8,12)
  model.train();x=torch.tensor([[2,5,6,10,3]])
  z=model.get_span_logits(x,torch.ones_like(x),torch.zeros_like(x),[torch.tensor([1,2])],[torch.tensor([1,2])])
  loss=weighted_loss(z,[torch.tensor([1,0])],[torch.tensor([.6,1.])],[torch.tensor([True,True])]);loss.backward()
  assert model.bert.embeddings.word_embeddings.weight.grad.abs().sum()>0
  # Changing a masked label must not alter either its loss contribution or normalization.
  logits=[torch.tensor([[2.,-1.,0.],[.5,1.5,-.5]],requires_grad=True)]
  ws=[torch.tensor([0.,1.])];masks=[torch.tensor([False,True])]
  a=weighted_loss(logits,[torch.tensor([1,0])],ws,masks)
  b=weighted_loss(logits,[torch.tensor([2,0])],ws,masks)
  assert torch.equal(a,b)
  gold=[{'text':'abcdefghij','entities':[{'start':0,'end':8,'entity_type':'taxa'}]}]
  predictions=[[{'start':0,'end':8,'type':'taxa'},{'start':1,'end':8,'type':'taxa'}]]
  result=evaluate_predictions(gold,predictions)
  assert result['strict']['tp']==1 and result['partial']['tp']==1 and result['partial']['fp']==1
  wrong=evaluate_predictions(gold,[[{'start':0,'end':8,'type':'location'}]])
  assert wrong['strict']['tp']==wrong['partial']['tp']==0
  for name in ['voters','aggregate','train','evaluate']:
   subprocess.run([sys.executable,str(ROOT/'src'/f'{name}.py'),'--help'],check=True,stdout=subprocess.DEVNULL)
 print('PASS: encoder gradients, masking, one-to-one/type matching, and CLI imports. No API requests or full training.')
if __name__=='__main__':main()
