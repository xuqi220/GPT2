
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

num_proc = 4
num_proc_load_dataset = 4

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)


if __name__ == '__main__':
    remote_name = "sample-10BT"
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train",num_proc=num_proc_load_dataset)

    #  create a test split
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 12324
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 12344
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token (<|endoftext|>), e.g. 50256 for gpt2 bpe 
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text', 'id', 'dump', 'url', 'file_path', 'language', 'language_score', 'token_count', 'score', 'int_score'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # split the ids in each dataset into multiple file we can use for training
    shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
    total_batches = 4096 # split the large file into 4096 batches
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")
        shard_idx = 0
        # init shard
        filename = os.path.join(os.path.dirname(__file__), f'{split}_{shard_idx}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(shard_size,))
        token_count = 0
        # write each batch into shard
        for batch_idx in tqdm(range(total_batches)):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            if token_count+len(arr_batch)<shard_size: # Write into current shard
                arr[token_count : token_count + len(arr_batch)] = arr_batch
                token_count += len(arr_batch)
            else:# next shard
                arr.flush()
                shard_idx+=1
                filename = os.path.join(os.path.dirname(__file__), f'{split}_{shard_idx}.bin')
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(shard_size,))
                token_count = 0

    # train.bin is ~18.5GB, val.bin ~9.5MB
    # train has ~10B tokens (9,949,090,040)9,949,090,040
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
