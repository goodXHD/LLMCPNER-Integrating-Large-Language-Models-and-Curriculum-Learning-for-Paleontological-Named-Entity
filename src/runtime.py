"""Runtime settings shared by final voter and main-model training."""
import hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer
import engine

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def configure_runtime(args):
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Training follows the BF16 CUDA protocol; a supported GPU is required.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    if not 0 < args.memory_fraction <= 1:raise ValueError('memory-fraction must be in (0,1]')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,args.device)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU support is required')
    original_collate,original_model=engine.collate_fn,engine.new_model
    def dynamic_collate(items):
        batch=original_collate(items)
        n=int(batch['attention_mask'].sum(1).max())
        for key in ['input_ids','attention_mask','token_type_ids']:batch[key]=batch[key][:,:n]
        return batch
    def checkpointed_model(*items,**kwargs):
        model=original_model(*items,**kwargs)
        model.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return model
    engine.collate_fn=dynamic_collate
    engine.new_model=checkpointed_model
