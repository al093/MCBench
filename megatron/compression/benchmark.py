from megatron.compression.fixpoint import compress_4bit, decompress_4bit
from megatron.compression.topk import encoder, decoder
import torch
import time
import statistics

device = torch.device("cuda:0")
activation = torch.rand([16, 512, 8192], dtype=torch.float16, device=device)
compress_int4_traced = torch.compile(compress_4bit, fullgraph=True)
decompress_int4_traced = torch.compile(decompress_4bit, fullgraph=True)

top_k = activation.size() // 3  # compression by 3x
compress_topk_traced = torch.compile(encoder, fullgraph=True)
decompress_topk_traced = torch.compile(decoder, fullgraph=True)

torch.cuda.synchronize()
warmup_iters = 10
times = []
for i in range(100+warmup_iters):
    start = time.time()
    outputs = compress_int4_traced(activation)
    outputs = decompress_int4_traced(*outputs)

    # outputs = compress_topk_traced(activation, top_k)
    # outputs = decoder(*outputs)

    torch.cuda.synchronize()
    end = time.time()
    if i >= warmup_iters:
        times.append(end-start)

print(f"Mean time: {statistics.mean(times)}")
print(f"Standard deviation: {statistics.stdev(times)}")
print(f"Max time: {max(times)}")
print(f"Min time: {min(times)}")
print(f"Median time: {statistics.median(times)}")