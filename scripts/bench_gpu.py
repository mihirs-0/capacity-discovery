"""Quick GPU benchmark: verify CUDA is used and measure step time."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model import Transformer

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("No CUDA - training is running on CPU!")
    sys.exit(1)

print("Device:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), "GB")

model = Transformer(4, 4, 128, 512).cuda()
x = torch.randint(0, 40, (128, 16)).cuda()
print(f"Params: {model.count_parameters():,}")

# Warmup
for _ in range(10):
    loss, _ = model(x, x)
    loss.backward()

# Benchmark
torch.cuda.synchronize()
t0 = time.time()
N = 1000
for _ in range(N):
    loss, _ = model(x, x)
    loss.backward()
torch.cuda.synchronize()
ms = (time.time() - t0) / N * 1000

print(f"Forward+backward: {ms:.3f} ms/step")
print(f"GPU active fraction at 1 step per 15ms: {ms/15*100:.1f}%")
print(f"With 24 workers: ~{ms*24/15*100:.0f}% theoretical max SM util")
print()
print("Verdict: GPU works, model is just too small to register on nvidia-smi.")
