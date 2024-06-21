# GPT-2

本项目是对GPT2（124M）模型结构的复现和训练

# 快速开始

* `model.py` GPT2模型结构。具体介绍请参考代码注释，下面根据个人情况记录一下细节：
    
1. **Token Embedding层和llm_head参数共享**
   
   Token Embedding 和 lm_head 分别是将token映射成为向量和将向量映射成为token, 他们的weight属性的shape=(vocab_size * n_embd)。nn.Linear() 的forward() 调用的是F.linear()函数，具体参考pytorch[官方文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear)。另外为什么共享？[依据在这里](https://arxiv.org/abs/1608.05859)论文还没看，看后补上
    
2. **模型参数初始化--方差**
   ```
   def init_weight(self, module):
        # init model weight
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
   ```
   上面是常用的模型初始化代码，但是GPT2采用了
   ```
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
    
   ```
   embedding层的参数正常初始化，然而对其他有FLAG标记的模块的方法差进行了scalling，为什么？
   ```
    import torch

    x = torch.zeros(768)
    n = 100
    for i in range(n):
        x += torch.randn(768)
    print(x.std())
    tensor(9.6580)
   ```
   我们发现tensor`x`经过多次残差连接后他的方差变大了！！！为了缓解这种现象，我们对其进行scalling,并选择了残差链接的次数作为缩放因子。
    ```
    import torch

    x = torch.zeros(768)
    n = 100
    for i in range(n):
        x += n**-0.5 * torch.randn(768)
    print(x.std())
    tensor(1.0784)
   ```
   最后采用的方差为 `std *= (2*self.config.n_layer)**-0.5`，2的原因是每个block使用了两个残差链接。

* `model_size.py`对模型参数量的估计
















# 致谢

非常感谢karpathy大神

https://github.com/karpathy/nanoGPT

https://github.com/karpathy/build-nanogpt
