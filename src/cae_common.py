import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

# FIXED SETTINGS (decided before any test result was seen; same for all classes)
SIZE = 256      # whole image resized to 256x256 (AITEX tiles are already 256)
PATCH = 128     # training patch size and inference window size
STRIDE = 32     # inference window stride
SIGMA = 4       # Gaussian smoothing of the residual map
LATENT = 100


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))


def load_img(path):
    im = Image.open(path).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


class CAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        def down(i, o): return [nn.Conv2d(i, o, 4, 2, 1), nn.LeakyReLU(0.2)]
        def up(i, o): return [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.LeakyReLU(0.2)]
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.enc = nn.Sequential(*down(3, 32), *down(32, 32), *down(32, 64), *down(64, 64), *down(64, 128))
        self.fc_e = nn.Linear(128 * 4 * 4, latent)
        self.fc_d = nn.Linear(latent, 128 * 4 * 4)
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.dec = nn.Sequential(*up(128, 64), *up(64, 64), *up(64, 32), *up(32, 32),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.fc_e(self.enc(x).flatten(1))
        return self.dec(self.fc_d(z).view(-1, 128, 4, 4))


def _gauss(win=11, sigma=1.5, device=None):
    g = torch.arange(win, dtype=torch.float32, device=device) - win // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(x, y):
    c = x.shape[1]
    w = _gauss(device=x.device)[None, None].repeat(c, 1, 1, 1)
    mx = F.conv2d(x, w, groups=c); my = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mx ** 2
    syy = F.conv2d(y * y, w, groups=c) - my ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mx * my
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return s.mean()


@torch.no_grad()
def anomaly_map(model, img_u8, device):
    """img_u8: HxWx3 uint8 (256x256). Returns 256x256 float32 raw anomaly map."""
    x = torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255).to(device)
    pos = list(range(0, SIZE - PATCH + 1, STRIDE))
    crops = [x[:, i:i + PATCH, j:j + PATCH] for i in pos for j in pos]
    rec = model(torch.stack(crops))
    acc = torch.zeros_like(x)
    cnt = torch.zeros(1, SIZE, SIZE, device=device)
    k = 0
    for i in pos:
        for j in pos:
            acc[:, i:i + PATCH, j:j + PATCH] += rec[k]
            cnt[:, i:i + PATCH, j:j + PATCH] += 1
            k += 1
    recon = acc / cnt
    res = (x - recon).abs().mean(0).cpu().numpy()
    return gaussian_filter(res, sigma=SIGMA).astype(np.float32)


def image_score(amap):
    return float(amap.max())   # rule chosen in advance: maximum of the map
