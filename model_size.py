from collections import OrderedDict
from dataclasses import dataclass

# model config
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


def model_size(config:GPTConfig):
    out = OrderedDict()
    # token embedding and token embedding
    out["embedding/token"] = config.vocab_size * config.n_embd
    out["embedding/position"] = config.block_size * config.n_embd
    out["embedding"] = out["embedding/token"]+out["embedding/position"]
    
    # attention in block
    out["attention/ln"] = config.n_embd
    out["attention/kqv"] = config.n_embd * 3*config.n_embd
    out["attention/proj"] = config.n_embd**2
    out['attention'] = out["attention/ln"]+out["attention/kqv"]+out["attention/proj"]
    
    # mlp in block
    out["mlp/ln"] = config.n_embd
    out["mlp/ffw"] = config.n_embd * 4*config.n_embd
    out["mlp/proj"] = 4*config.n_embd * config.n_embd
    out["mlp"] = out["mlp/ln"]+out["mlp/ffw"]+out["mlp/proj"]
    
    # size of block
    out["block"] = out["attention"]+out["mlp"]
    
    # size of transformer
    out["transformer"] = config.n_layer*out["block"]
    # size of last layernorm
    out["ln"] = config.n_embd
    # size of the lm_head
    out["lm_head"] = 0 # sharing with the embedding/tokens
    
    # total
    out["total"] = out["embedding"]+out["transformer"]+out["ln"]+out["lm_head"]

    return out

# compare our param count to that reported by PyTorch
ms = model_size(GPTConfig())
params_total = ms['total']
print(f"we see: {params_total}, expected: {124337664}, match: {params_total == 124337664}")
# create a header
print(f"{'name':20s} {'params':10s} {'ratio (%)':10s}")
for k,v in ms.items():
    print(f"{k:20s} {v:10d} {v/params_total*100:10.4f}")