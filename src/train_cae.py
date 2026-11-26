import argparse, json, os, time
import numpy as np, torch
from cae_common import *

p = argparse.ArgumentParser()
p.add_argument("--category", required=True)
p.add_argument("--data", default="MVTec_split")
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

seed_all(a.seed)
dev = "cuda"
folder = os.path.expanduser(f"~/datasets/{a.data}/{a.category}/train_good")
names = list_images(folder)
imgs = np.stack([load_img(f"{folder}/{n}") for n in names])
data = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(dev)
print("train images:", len(names), "from", folder, flush=True)

model = CAE().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
out_dir = os.path.expanduser(f"~/results/cae_{a.category}")
os.makedirs(out_dir, exist_ok=True)

t0 = time.time()
for step in range(1, a.steps + 1):
    idx = torch.randint(0, len(data), (a.batch,)).tolist()
    ys = torch.randint(0, SIZE - PATCH + 1, (a.batch,)).tolist()
    xs = torch.randint(0, SIZE - PATCH + 1, (a.batch,)).tolist()
    x = torch.stack([data[i, :, y:y + PATCH, xx:xx + PATCH] for i, y, xx in zip(idx, ys, xs)])
    x = x.float().div(255)
    rec = model(x)
    l1 = (rec - x).abs().mean()
    s = ssim(rec, x)
    loss = l1 + (1 - s)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 1:
        print(f"step {step} loss {loss.item():.4f} l1 {l1.item():.4f} ssim {s.item():.4f} t {time.time()-t0:.0f}s", flush=True)

ckpt = f"{out_dir}/cae_{a.category}.pt"
meta = {"category": a.category, "data": a.data, "steps": a.steps, "batch": a.batch, "lr": a.lr,
        "seed": a.seed, "size": SIZE, "patch": PATCH, "stride": STRIDE, "sigma": SIGMA,
        "latent": LATENT, "n_train": len(names)}
torch.save({"state_dict": model.state_dict(), **meta}, ckpt)
print("saved", ckpt, "fit seconds", round(time.time() - t0), flush=True)
