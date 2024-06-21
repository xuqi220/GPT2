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


def num_of_params(config:GPTConfig):
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

def model_size(params_total):
    # 假设参数是以fp32存储
    params_bytes = 4*params_total
    # 利用Adamw作为优化器(为每个参数保存了两份统计参数)
    params_and_buffers_bytes = params_bytes + 2*params_bytes
    print(f"est checkpoint size: {params_and_buffers_bytes/1e9:.2f} GB")
    gpu_memory = 40e9 # 40 GB A100 GPU, roughly
    print(f"memory ratio taken up just for parameters: {params_and_buffers_bytes / gpu_memory * 100:.2f}%")

def est_FLOPs():
    pass


# compare our param count to that reported by PyTorch
ms = num_of_params(GPTConfig())
params_total = ms['total']
print(f"we see: {params_total}, expected: {124337664}, match: {params_total == 124337664}")
# create a header
print(f"{'name':20s} {'params':10s} {'ratio (%)':10s}")
for k,v in ms.items():
    print(f"{k:20s} {v:10d} {v/params_total*100:10.4f}")
model_size(params_total)

# 注意上述计算方式忽略了中间过程变量，在实际使用中，模型占用的内存比这个要多，因为pytorch要维护一个计算图，记录中间过程，
# 这些中间过程将用于反向传播过程，比如gradient-checkpoint技术就是通过优化中间过程变量来减少内存占用的

