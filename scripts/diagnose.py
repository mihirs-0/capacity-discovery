"""Diagnose which operation is slow/hanging."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Step 1: basic matmul
print("\n[1] Basic matmul...", end=" ", flush=True)
x = torch.randn(128, 128).cuda()
w = torch.randn(128, 128).cuda()
torch.cuda.synchronize()
t = time.time()
for _ in range(100):
    y = x @ w
torch.cuda.synchronize()
print(f"{(time.time()-t)*1000/100:.3f} ms/op")

# Step 2: model forward
print("[2] Model forward...", end=" ", flush=True)
from src.model import Transformer
model = Transformer(4, 4, 128, 512).cuda()
batch = torch.randint(0, 40, (128, 16)).cuda()
torch.cuda.synchronize()
t = time.time()
loss, _ = model(batch, batch)
torch.cuda.synchronize()
print(f"{time.time()-t:.3f}s")

# Step 3: model backward
print("[3] Backward...", end=" ", flush=True)
t = time.time()
loss.backward()
torch.cuda.synchronize()
print(f"{time.time()-t:.3f}s")

# Step 4: dataset creation
print("[4] Dataset (D=1000)...", end=" ", flush=True)
from src.task import SurjectiveMap
t = time.time()
ds = SurjectiveMap(K=20, n_b=50, seed=42)
print(f"{time.time()-t:.3f}s")

# Step 5: compute_train_loss
print("[5] compute_train_loss...", end=" ", flush=True)
from src.diagnostics import compute_train_loss
t = time.time()
r = compute_train_loss(model, ds.get_full(), torch.device("cuda"))
print(f"{time.time()-t:.3f}s  loss={r['train_loss']:.4f}")

# Step 6: compute_z_shuffle_gap
print("[6] compute_z_shuffle_gap...", end=" ", flush=True)
from src.diagnostics import compute_z_shuffle_gap
t = time.time()
r = compute_z_shuffle_gap(model, ds.get_full(), torch.device("cuda"))
print(f"{time.time()-t:.3f}s  gap={r['z_shuffle_gap']:.4f}")

# Step 7: compute_group_accuracy
print("[7] compute_group_accuracy...", end=" ", flush=True)
from src.diagnostics import compute_group_accuracy
t = time.time()
r = compute_group_accuracy(model, ds, torch.device("cuda"), n_groups=50)
print(f"{time.time()-t:.3f}s  acc={r['group_accuracy_mean']:.4f}")

# Step 8: compute_stable_ranks
print("[8] compute_stable_ranks...", end=" ", flush=True)
from src.diagnostics import compute_stable_ranks
t = time.time()
r = compute_stable_ranks(model)
print(f"{time.time()-t:.3f}s")

print("\nAll diagnostics OK. No hang detected.")
