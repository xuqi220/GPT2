from dataclasses import dataclass
import math
import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F

# model config
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
class CasualSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        assert self.config.n_embd % self.config.n_head == 0
        self.c_attn = nn.Linear(self.config.n_embd, 3*self.config.n_embd)
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
    
    def forward(self, x):
        # batch_size, Sequence length, embedding_size
        B,T,C = x.shape 
        # transferred by W_q, W_k, W_v
        qkv = self.c_attn(x)
        # get qkv
        q,k,v = qkv.split(self.n_embd,dim=2)
        # multi-head qkv
        q = q.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        k = k.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        v = v.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        # # attention
        # attn = (q @ k.transpose(-2, -1))*(1.0/math.sqrt(q.shape[-1]))
        # # mask previous tokens
        # attn = attn.masked_fill(self.bias[:,:,T,T]==0, float("-inf"))
        # # cal attn score 
        # attn = F.softmax(attn, dim=-1)
        # # cal output
        # y = attn @ v
        # 上面计算attention的代码等价于scaled_dot_product_attention函数
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y
        
class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        self.c_fc = nn.Linear(self.config.n_embd, 4*self.config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*self.config.n_embd, self.config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class block(nn.Module):
    def __init__(self, config:GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.attn = CasualSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(self.config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd), # 词向量
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd), # 位置向量
            h = nn.ModuleList([block(self.config) for _ in range(self.config.n_layer)]), # transformer blocks
            ln_f = nn.LayerNorm(self.config.n_embd) # layerNormal
            
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
    
    def forward(self,idx):
        B,T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
        
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



model = GPT.from_pretrained("gpt2")
model.eval()
model.to("cuda")
        


