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
        self.FLAG = 1
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(self.config.block_size, self.config.block_size)).view(1,1, self.config.block_size,self.config.block_size)
        )
    
    def forward(self, x):
        # batch_size, Sequence length, embedding_size
        B,T,C = x.shape 
        # transferred by W_q, W_k, W_v (B, T, 3*C)
        qkv = self.c_attn(x)
        # get qkv (B, T, C)
        q,k,v = qkv.split(self.config.n_embd,dim=2)
        # multi-head qkv ()
        q = q.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        k = k.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        v = v.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        # attention (B, n_head, T, T)
        attn = (q @ k.transpose(-2, -1))*(1.0/math.sqrt(q.shape[-1]))
        # mask previous tokens
        attn = attn.masked_fill(self.bias[:,:,T,T]==0, float("-inf"))
        # cal attn score (B, n_head, T, T)
        attn = F.softmax(attn, dim=-1)
        # cal output (B, n_head, T, C//n_head)
        y = attn @ v
        # 上面计算attention的代码等价于scaled_dot_product_attention函数
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # (B, T, C)
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
        self.FLAG = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class block(nn.Module):
    # each block consists of layernormal, attention, MLP
    def __init__(self, config:GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.attn = CasualSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(self.config)
    
    def forward(self,x):
        # residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            # 词向量
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd), 
            # 位置向量
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd), 
            # transformer blocks
            h = nn.ModuleList([block(self.config) for _ in range(self.config.n_layer)]), 
            # layerNormal
            ln_f = nn.LayerNorm(self.config.n_embd) 
            
        ))
        # pred token
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
        # weight share scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init weight
        # Apply fn(init_weight) recursively to every submodule
        self.apply(self.init_weight)
    
    def init_weight(self, module):
        # init model weight
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "FLAG"):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self,idx, targets=None):
        B,T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
           
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
    
class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B # batch size
        self.T = T # sequence length
        # get tokens for train
        with open("data.txt","r",encoding="utf-8") as fi:
            text = fi.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # sample tokens
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T
        if self.current_position>len(self.tokens):
            self.current_position=0
        return x, y


if __name__=="__main__":
    # 检测可用设备
    if torch.cuda.is_available():  
        device = "cuda:0" 
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # init model from huggingface
    # model = GPT.from_pretrained("gpt2")
    
    # random init
    model = GPT(GPTConfig())
    # train model
    model.to(device)
    model.train()
    train_loader = DataLoaderLite(B=5, T=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(100):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss {loss.item()}")
            

    import sys; sys.exit(0)
#--------------------------------inference-----------------------------------
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    num_return_sequence = 5
    max_length_sequence = 30

    tokens =enc.encode("hello")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
    x = tokens.to(device)

    while x.size(1)<max_length_sequence:
        with torch.no_grad():
            # 获取 next tokens logits
            logits, loss = model(x) # (B,T,Vocab_size)
            logits = logits[:,-1,:] # (B, Vocab_size)
            # 归一化
            probs = F.softmax(logits, dim=-1)
            # 从top 50 的候选token中采样
            top_probs, top_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(top_probs, 1)
            xcol = torch.gather(top_indices, -1, ix)
            # 将预测的token和previous tokens拼接作为下一次输入
            x = torch.cat((x, xcol), dim=1)
    # 解码
    for i in range(num_return_sequence):
        tokens = x[i, :max_length_sequence].tolist()
        text = enc.decode(tokens)
        print(f">{text}")

