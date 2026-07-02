import os, math, random, time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import CosineAnnealingLR


default_device = os.getenv("TORCH_DEVICE", "").strip()

print("TORCH_DEVICE env:", os.getenv("TORCH_DEVICE"), flush=True)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
print("torch.cuda.device_count():", torch.cuda.device_count(), flush=True)


if default_device:
    device = torch.device(default_device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda" and not torch.cuda.is_available():
    print("[warn] CUDA non disponibile: passo a CPU", flush=True)
    device = torch.device("cpu")


if device.type == "cuda":
    try:
        torch.cuda.set_device(0)
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")
        print("GPU OK:", torch.cuda.get_device_name(0), flush=True)

    except Exception as e:
        raise RuntimeError(
            "CUDA visibile ma non utilizzabile (busy/unavailable). "
            "Rilancia il job o forza un nodo diverso (es. gnode13)."
        ) from e

print("device:", device, flush=True)


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(42)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


"""Global hyperparam"""
L = 1000
TIME_ENCODING_SIZE = 128         
COND_DIM = 8                     
IMAGE_SHAPE = (3, 64, 64)        

LR = 2e-4
EPOCHS = 500
BATCH_SIZE = 128

P_DROP = 0.2

OUT_DIR = "output_dir"
ensure_dir(OUT_DIR)
ensure_dir(os.path.join(OUT_DIR, "ckpt"))
ensure_dir(os.path.join(OUT_DIR, "samples"))
print("OUT_DIR:", OUT_DIR)


"""Dataset wrapper"""
try:
    from torchvision.transforms import v2 as T  
    def build_transform(image_size: int):
        return T.Compose([
            T.ToImage(),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
except Exception:
    from torchvision import transforms as T
    def build_transform(image_size: int):
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])


DATA_ROOT = "/home/pfoggia/GenerativeAI/CELEBA"
DOWNLOAD = False

transform = build_transform(64)

base_ds = CelebA(
    root=DATA_ROOT,
    split="train",
    target_type="attr",
    transform=transform,
    download=False
)

MALE_IDX, SMILE_IDX, YOUNG_IDX = 20, 31, 39

def attrs_to_class(attr: torch.Tensor) -> torch.Tensor:
    male    = (attr[:, MALE_IDX]  == 1).long()
    smiling = (attr[:, SMILE_IDX] == 1).long()
    young   = (attr[:, YOUNG_IDX] == 1).long()
    return male*4 + smiling*2 + young  # 0..7


class CelebACond(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        x, attr = self.ds[i]                
        # attrs_to_class si aspetta (B,40), quindi faccio unsqueeze e poi prendo [0]
        cls = attrs_to_class(attr.unsqueeze(0))[0].item()
        return x, cls

ds = CelebACond(base_ds)

dataloader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=(device.type == "cuda"),
    drop_last=True
)

cond_one_hot = torch.eye(8, device=device)

x, target = next(iter(dataloader))
print("x:", x.shape, "target:", target.shape, "min/max:", target.min().item(), target.max().item())


"""NoiseSchedule + TimeEncoding"""
class NoiseSchedule:
    def __init__(self, L, s=0.008, device=device):
        self.L = L
        t = torch.linspace(0.0, L, L+1, device=device) / L
        a = torch.cos((t+s)/(1+s)*torch.pi/2)**2
        a = a / a[0]
        self.beta = (1 - a[1:] / a[:-1]).clamp(0.0, 0.99)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0)
        self.one_minus_beta = 1 - self.beta
        self.one_minus_alpha = 1 - self.alpha
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.sqrt_1_alpha = torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta = torch.sqrt(self.one_minus_beta)

    def __len__(self):
        return self.L


class TimeEncoding:
    def __init__(self, L, dim, device=device):
        self.L = L
        self.dim = dim
        dim2 = dim // 2
        encoding = torch.zeros(L, dim)
        ang = torch.linspace(0.0, torch.pi/2, L)
        logmul = torch.linspace(0.0, math.log(40), dim2)
        mul = torch.exp(logmul)
        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2*i]   = torch.sin(a)
            encoding[:, 2*i+1] = torch.cos(a)
        self.encoding = encoding.to(device=device)

    def __len__(self):
        return self.L

    def __getitem__(self, t):
        return self.encoding[t]

noise_schedule = NoiseSchedule(L)
time_encoding = TimeEncoding(L, TIME_ENCODING_SIZE)

"""UNetBlock + Network"""
class UNetBlock(nn.Module):
    def __init__(self, size, outer_features, inner_features, cond_features, inner=None):
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.cond_features = cond_features
        self.inner = inner

        self.encoder = self.build_encoder(outer_features + cond_features, inner_features)
        self.decoder = self.build_decoder(inner_features + cond_features + TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2 * outer_features, outer_features)

    def forward(self, x, time_encodings, cond):
        x0 = x

        # encoder input: concat cond (broadcast)
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        y0 = torch.cat((x, cc), dim=1)
        y = self.encoder(y0)

        if self.inner is not None:
            y = self.inner(y, time_encodings, cond)

        half = self.size // 2
        cc2 = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, half, half)
        tt = time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half, half)

        # decoder input: concat (y, cond, time)
        y1 = torch.cat((y, cc2, tt), dim=1)
        x1 = self.decoder(y1)

        # skip + 1x1 combiner
        x2 = torch.cat((x1, x0), dim=1)
        return self.combiner(x2)

    def build_combiner(self, from_features, to_features):
        return nn.Conv2d(from_features, to_features, 1)

    def build_encoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(from_features),
            nn.SiLU(),
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.SiLU(),
        )

    def build_decoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(from_features),
            nn.SiLU(),
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.SiLU(),
        )

class Network(nn.Module):
    def __init__(self, image_shape, feature_list, cond_dim):
        super().__init__()
        self.image_shape = image_shape
        self.cond_dim = cond_dim

        C, H, W = image_shape
        self.pre = nn.Sequential(
            nn.Conv2d(C, feature_list[0], 3, padding=1),
            nn.SiLU()
        )

        self.unet = self.build_unet(H, feature_list, cond_dim)

        self.post = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(feature_list[0], C, 3, padding=1)
        )

    def build_unet(self, size, feature_list, cond_dim):
        if len(feature_list) > 2:
            inner = self.build_unet(size // 2, feature_list[1:], cond_dim)
        else:
            inner = None
        return UNetBlock(size, feature_list[0], feature_list[1], cond_dim, inner)

    def forward(self, x, t, cond):
        time_enc = time_encoding[t]
        h = self.pre(x)
        h = self.unet(h, time_enc, cond)
        return self.post(h)

FEATURES = [64, 128, 256, 512]
model = Network(IMAGE_SHAPE, FEATURES, COND_DIM).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

RESUME_PATH = "/home/M.APUZZO13/runs/ddpm_celeba64_ex4style_cosine_500/ckpt/ckpt_latest.pt"
start_epoch = 0

if os.path.exists(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"]
    print(f"[resume] loaded checkpoint from epoch {start_epoch}", flush=True)
else:
    print("[resume] no checkpoint found, starting from scratch", flush=True)

print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

"""training_epoch"""
epoch_count = 0
def training_epoch(dataloader):
    global epoch_count
    model.train()
    average_loss = 0.0

    num_batches = len(dataloader)

    for b, (x, target) in enumerate(dataloader, start=1):
        x = x.to(device=device)
        target = target.to(device=device)
        n = x.shape[0]

        cond = cond_one_hot[target].clone()  

        # classifier-free drop conditioning with prob P_DROP
        u = torch.rand((n,), device=device)
        cond[u < P_DROP, :] = 0.0

        # random steps
        t = torch.randint(0, L, (n,), device=device)

        # forward diffusion
        eps = torch.randn_like(x)
        sqrt_alpha   = noise_schedule.sqrt_alpha[t].view(n, 1, 1, 1)
        sqrt_1_alpha = noise_schedule.sqrt_1_alpha[t].view(n, 1, 1, 1)
        zt = sqrt_alpha * x + sqrt_1_alpha * eps

        # predict eps and optimize
        g = model(zt, t, cond)
        loss = loss_function(g, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        average_loss = 0.9 * average_loss + 0.1 * loss.detach().cpu().item()

        # --- progress within epoch ---
        if (b % 50 == 0) or (b == 1) or (b == num_batches):
            pct = 100.0 * b / num_batches
            print(f"  batch {b:4d}/{num_batches} ({pct:5.1f}%)  loss(EMA)={average_loss:.6f}")

    epoch_count += 1
    print(f" Epoch {epoch_count} completed. loss(EMA)={average_loss:.6f}")
    return average_loss

"""Sampling: generate + ddim_step + generate_ddim"""
@torch.no_grad()
def generate(cond, lam):
    n = cond.shape[0]
    z = torch.randn(n, *IMAGE_SHAPE, device=device)
    cond0 = torch.zeros_like(cond)
    model.eval()

    for kt in reversed(range(L)):
        t = torch.tensor(kt, device=device).view(1).expand(n)

        beta = noise_schedule.beta[kt]
        sqrt_1_alpha = noise_schedule.sqrt_1_alpha[kt]
        sqrt_1_beta  = noise_schedule.sqrt_1_beta[kt]
        sqrt_beta    = noise_schedule.sqrt_beta[kt]

        g1 = model(z, t, cond)
        g0 = model(z, t, cond0)
        g  = lam * g1 + (1 - lam) * g0

        mu = (z - beta / sqrt_1_alpha * g) / sqrt_1_beta

        if kt > 0:
            eps = torch.randn_like(z)
            z = mu + sqrt_beta * eps
        else:
            z = mu
    return z

def ddim_step(zt, g, eta, tau_curr, tau_prev):
    a_curr = noise_schedule.alpha[tau_curr]
    a_prev = noise_schedule.alpha[tau_prev] if tau_prev >= 0 else 1.0
    sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_curr) * (1.0 - a_curr / a_prev))
    c1 = torch.sqrt(a_prev / a_curr)
    c2 = torch.sqrt(1.0 - a_prev - sigma**2) - torch.sqrt(a_prev * (1.0 - a_curr) / a_curr)
    eps = torch.randn_like(zt)
    z_prev = c1 * zt + c2 * g + sigma * eps
    return z_prev

def lam_schedule(progress: float, lam_min: float, lam_max: float, gamma: float = 2.0) -> float:
    """
    progress in [0,1]: 0 = inizio reverse (molto rumore), 1 = fine reverse (poco rumore)
    """
    return lam_min + (lam_max - lam_min) * (progress ** gamma)

@torch.no_grad()
def generate_ddim(cond, lam, tau, eta=0.25, lam_min=1.3, gamma=2.3):
    n = cond.shape[0]
    z = torch.randn(n, *IMAGE_SHAPE, device=device)
    cond0 = torch.zeros_like(cond)
    model.eval()

    T = len(tau)
    denom = max(1, T - 1)

    for kt in reversed(range(T)):
        tau_curr = tau[kt]
        tau_prev = tau[kt - 1] if kt > 0 else -1
        t = torch.tensor(tau_curr, device=device).view(1).expand(n)

        # progress: 0 (molto rumore) -> 1 (poco rumore)
        progress = (T - 1 - kt) / denom
        lam_t = lam_schedule(progress, lam_min=lam_min, lam_max=lam, gamma=gamma)

        g1 = model(z, t, cond)
        g0 = model(z, t, cond0)

        g = lam_t * g1 + (1.0 - lam_t) * g0
        z = ddim_step(z, g, eta, tau_curr, tau_prev)

    return z


"""Preview"""
def save_ddim_grid(epoch, k_per_class=4, lam=2.3, stride=4, eta=0.25, lam_min=1.3, gamma=2.3):
    tau = list(range(0, L, stride))
    cls = torch.arange(0, COND_DIM, device=device).repeat_interleave(k_per_class)
    cond = cond_one_hot[cls]

    x = generate_ddim(cond, lam=lam, tau=tau, eta=eta, lam_min=lam_min, gamma=gamma)
    x = (x.clamp(-1, 1) + 1) / 2
    grid = make_grid(x, nrow=k_per_class, padding=2)

    # eta in stringa "safe" per filename (0.5 -> "0p5")
    eta_tag = str(eta).replace(".", "p")

    out_path = os.path.join(
        OUT_DIR, "samples",
        f"samples_ddim_e{epoch:03d}_lam{lam:.2f}_stride{stride}_eta{eta_tag}_N{len(tau)}.png"
    )
    save_image(grid, out_path)
    print("saved:", out_path)

"""Train"""
def should_save_ckpt(epoch: int, total_epochs: int) -> bool:
    if epoch == total_epochs:
        return True
    if epoch <= 50:
        return (epoch % 5) == 0
    return (epoch % 10) == 0

def save_checkpoint(epoch: int, model: nn.Module, optimizer, scheduler, out_dir: str):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    ckpt_path = os.path.join(out_dir, "ckpt", f"ckpt_e{epoch:03d}.pt")
    torch.save(state, ckpt_path)
    print("saved ckpt:", ckpt_path, flush=True)

    latest_path = os.path.join(out_dir, "ckpt", "ckpt_latest.pt")
    torch.save(state, latest_path)

loss_history = []

for i in range(start_epoch, EPOCHS):
    epoch = i + 1

    print(f"\n==============================")
    print(f" START epoch {epoch}/{EPOCHS}")
    print(f"==============================")

    avg_loss = training_epoch(dataloader)
    loss_history.append(avg_loss)

    scheduler.step()
    print(f"[lr] current lr -> {optimizer.param_groups[0]['lr']:.6e}", flush=True)

    if should_save_ckpt(epoch, EPOCHS):
        save_checkpoint(epoch, model, optimizer, scheduler, OUT_DIR)

    if (epoch % 5 == 0) or (epoch == EPOCHS):
        save_ddim_grid(epoch, k_per_class=4, lam=2.3, stride=4, eta=0.25)

    print(f" END epoch {epoch}/{EPOCHS}  loss(EMA)={avg_loss:.6f}")