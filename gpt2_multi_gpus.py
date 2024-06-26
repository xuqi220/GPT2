import os
import time
import math
import torch
import inspect
import tiktoken
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# model config
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
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
        
        # # attention (B, n_head, T, T)
        # attn = (q @ k.transpose(-2, -1))*(1.0/math.sqrt(q.shape[-1]))
        # # mask previous tokens
        # attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float("-inf"))
        # # cal attn score (B, n_head, T, T)
        # attn = F.softmax(attn, dim=-1)
        # # cal output (B, n_head, T, C//n_head)
        # y = attn @ v
        
        # 上面计算attention的代码等价于scaled_dot_product_attention函数
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.memmap(filename, dtype=np.uint16, mode='r')
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_process, data_root, split):
        assert split in ["train", "val"]
        self.B = B # batch size
        self.T = T # sequence length
        self.process_rank = process_rank # 当前进程号
        self.num_process = num_process # 进程数
        shards = sorted([s for s in os.listdir(data_root) if split in s])
        shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards)>0, f"no shards found for {split} split"
        self.shards = shards
        # 当前进程采样起始位置
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        if process_rank == 0:
            print(f"found {len(self.shards)} for {split} split")
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
            
    def next_batch(self):
        B, T = self.B, self.T
        # sample tokens
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T*self.num_process
        # 越界后重置初始值
        if self.current_position+(B*T*self.num_process+1)>len(self.tokens):
            self.current_shard = (self.current_shard+1)%len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position=self.B * self.T * self.process_rank
        return x, y
    
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

if __name__=="__main__":
    import os
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    # 设置DDP环境
    # torchrun 命令将world size，local rank， rank 等信息设置在环境变量中
    ddp = int(os.environ.get('RANK',-1)) != -1 # 若'RANK'不在环境变量中，说明未初始化分布式环境，返回默认值-1
    if ddp:# 分布式环境
        assert torch.cuda.is_available()
        # 初始化进程组
        init_process_group('nccl') 
        # 获取分布式环境变量
        ddp_rank = int(os.environ.get('RANK'))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
        ddp_world_size = int(os.environ.get('WORLD_SIZE'))
        # 设置当前进程对应的设备
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank==0
    else: # 非分布式环境
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        ddp_rank = 1
        master_process = True
        # 检测可用设备
        device = "cpu"
        if torch.cuda.is_available():  
            device = "cuda" 
        elif torch.backends.mps.is_available():
            device = "mps"
    print(f"use device: {device}")        
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    # 设置随机种子，保证每个设备上的模型一致
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # 利用TF32加速训练
    torch.set_float32_matmul_precision("high")
    
    # 梯度累积(需要考虑分布式)
    total_batch_size = 524288 # 0.5M tokens based on GPT-3 Papers
    B, T = 16, 1024
    total_batch_size%(B*T*ddp_world_size) == 0
    grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"calculated gradient accumulation steps per process: {grad_accum_steps}")
    
    # 数据集加载（需要考虑分布式环境）
    train_loader = DataLoaderLite(B=B, 
                                  T=T, 
                                  process_rank=ddp_rank, 
                                  num_process=ddp_world_size,
                                  data_root="tokenized_finewebedu",
                                  split="train")
    
    val_loader = DataLoaderLite(B=B, 
                                T=T, 
                                process_rank=ddp_rank, 
                                num_process=ddp_world_size,
                                data_root="tokenized_finewebedu",
                                split="val")
    # random init
    model = GPT(GPTConfig())
    model.to(device)      
    
    # 编译模型，优化内存读写，加速训练
    use_compile = False # torch.compile interferes and Generation.
    if use_compile:
        model = torch.compile(model)
    
    # ddp warp model
    if ddp:
        # 梯度同步：
        # pytorch 提供的DDP 在反向传播时首先单独计算每个gpu_rank上的梯度；（此时每个设备上各自保留着一份不一样的梯度）
        # 然后，多个设备相互通信，所有设备上的梯度聚合在一起求平均。（reduce）
        # 最后，将平均后的梯度分配到每个gpu上，此时每个GPU保留同样的梯度，并各自更新梯度。(map)
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    
    max_lr = 6e-4
    min_lr = 0.1*max_lr
    warmup_steps = 715
    max_steps = 19073
    def get_lr(step): 
        # 前半段是线性增加的
        if step<warmup_steps:
            return max_lr * (step+1)/warmup_steps
        # 超过max_steps取最小
        if step>max_steps:
            return min_lr
        # 中间过程是呈余弦曲线衰减直到min_lr
        decay_ratio = (step-warmup_steps)/(max_steps-warmup_steps) # 衰减率越来越大
        assert 0<=decay_ratio<=1
        coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio)) # 衰减系数越来越小
        return min_lr+coeff*(max_lr-min_lr)
    
    
    # 将参数分组采用weight decay，配置优化器
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
    
    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    
    iteration_step = 250
    for step in range(max_steps):
        t0 = time.time()
        last_step = (max_steps-1)==step       
         
        # evaluate model
        if step%iteration_step==0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)

 
        # do generation
        if ((step>0 and step%iteration_step==0) or last_step) and not use_compile:# 
            model.eval()
            num_return_sequence = 5
            max_length = 30

            tokens =enc.encode("hello i am a lanuguage model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1)<max_length:
                with torch.no_grad():
                    # 获取 next tokens logits
                    logits, loss = model(xgen) # (B,T,Vocab_size)
                    logits = logits[:,-1,:] # (B, Vocab_size)
                    # 归一化
                    probs = F.softmax(logits, dim=-1)
                    # 从top 50 的候选token中采样
                    top_probs, top_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(top_probs, 1, generator=sample_rng)# (B, 1)
                    xcol = torch.gather(top_indices, -1, ix) # (B, 1)
                    # 将预测的token和previous tokens拼接作为下一次输入
                    xgen = torch.cat((xgen, xcol), dim=1)
            # 解码
            for i in range(num_return_sequence):
                tokens = xgen[i, :max_length].tolist()
                text = enc.decode(tokens)
                print(f"\nrank-{ddp_rank}-gpt-2 >>> {text}")
        
        # train model 
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # 混合精度训练，因为采用了bfloat16（表示范围与FP32一致）所以不需要梯度缩放
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                # import code; code.interact(local=locals())
            # 由于是loss.backward()是梯度累加，所以先reduced by mean
            # cross_entropy采用已经在mini_batch内取了均值所以这里只用除以梯度累积的次数
            loss = loss/grad_accum_steps 
            loss_accum += loss.detach()
            # 在默认的分布式训练的环境下，loss.backward()操作执行了allduce和map
            # 然而，在梯度累积阶段我们行不想直接进行梯度同步，我们希望在每个梯度累积结束的时候进行梯度同步
            # pytorch 提供了一个上下文管理器DDP.no_sync(),他的但是调用起来代码结构不好看
            # 我们遵循他的源码实现，采用了require_backward_grad_sync属性
            if ddp:
                # 在梯度累积的最后一步，model的该属性值为True，backward操作才会执行梯度同步
                model.require_backward_grad_sync = (micro_step==grad_accum_steps-1)
            # 到达accumulate bundary前每个节点独立计算梯度
            # 到达accumulate bundary梯度同步
            loss.backward() 
        # loss_accum同步
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # 梯度裁剪
        lr = get_lr(step) # 动态学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (total_batch_size)/(t1-t0)
        if master_process:
            print(f"step: {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    if ddp:
        destroy_process_group()            

    

