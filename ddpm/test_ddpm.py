import os
import math
import torch
import matplotlib.pyplot as plt
from torch import nn

# =========================
# Config
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 1000
TIME_ENCODING_SIZE = 128
COND_DIM = 8
IMAGE_SHAPE = (3, 64, 64)
FEATURES = [64, 128, 256, 512]

BATCH_SIZE = 3
LAM = 2.3

CHECKPOINT_PATH = "/home/M.APUZZO13/runs/ddpm_celeba64_ex4style_cosine_500/ckpt/ckpt_e500.pt"
OUTPUT_DIR = "/home/M.APUZZO13/runs/ddpm_celeba64_ex4style_cosine_500/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Schedule + TimeEncoding
# =========================
class NoiseSchedule:
    def __init__(self, L, s=0.008, device=device):
        self.L = L
        t = torch.linspace(0.0, L, L + 1, device=device) / L
        a = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
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
        ang = torch.linspace(0.0, torch.pi / 2, L)
        logmul = torch.linspace(0.0, math.log(40), dim2)
        mul = torch.exp(logmul)
        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2 * i] = torch.sin(a)
            encoding[:, 2 * i + 1] = torch.cos(a)
        self.encoding = encoding.to(device=device)

    def __len__(self):
        return self.L

    def __getitem__(self, t):
        return self.encoding[t]


noise_schedule = NoiseSchedule(L)
time_encoding = TimeEncoding(L, TIME_ENCODING_SIZE)

# =========================
# Model
# =========================
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

        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        y0 = torch.cat((x, cc), dim=1)
        y = self.encoder(y0)

        if self.inner is not None:
            y = self.inner(y, time_encodings, cond)

        half = self.size // 2
        cc2 = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, half, half)
        tt = time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half, half)

        y1 = torch.cat((y, cc2, tt), dim=1)
        x1 = self.decoder(y1)

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

# =========================
# Sampling
# =========================
@torch.no_grad()
def generate(model, cond, lam):
    n = cond.shape[0]
    z = torch.randn(n, *IMAGE_SHAPE, device=device)
    cond0 = torch.zeros_like(cond)
    model.eval()

    for kt in reversed(range(L)):
        t = torch.tensor(kt, device=device).view(1).expand(n)

        beta = noise_schedule.beta[kt]
        sqrt_1_alpha = noise_schedule.sqrt_1_alpha[kt]
        sqrt_1_beta = noise_schedule.sqrt_1_beta[kt]
        sqrt_beta = noise_schedule.sqrt_beta[kt]

        g1 = model(z, t, cond)
        g0 = model(z, t, cond0)
        g = lam * g1 + (1 - lam) * g0

        mu = (z - beta / sqrt_1_alpha * g) / sqrt_1_beta

        if kt > 0:
            eps = torch.randn_like(z)
            z = mu + sqrt_beta * eps
        else:
            z = mu

    return z

# =========================
# Load checkpoint
# =========================
model = Network(IMAGE_SHAPE, FEATURES, COND_DIM).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

cond_one_hot = torch.eye(COND_DIM, device=device)

# =========================
# Grid generation
# =========================
def generate_condition_combinations_and_save(model, checkpoint, batch_size=3, lam=2.3):
    all_conditions = [
        (m, s, y)
        for m in [0, 1]
        for s in [0, 1]
        for y in [0, 1]
    ]

    num_rows = len(all_conditions)
    num_cols_total = 1 + batch_size

    fig, axes = plt.subplots(
        num_rows,
        num_cols_total,
        figsize=(2.6 * num_cols_total, 2.3 * num_rows)
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    with torch.no_grad():
        for row_idx, (m, s, y) in enumerate(all_conditions):
            cls = m * 4 + s * 2 + y
            cond = cond_one_hot[cls].unsqueeze(0).repeat(batch_size, 1)

            gen = generate(model, cond, lam=lam)

            # da [-1,1] a [0,1]
            gen = (gen.clamp(-1, 1) + 1) / 2

            gender_txt = "Maschio" if m == 1 else "Femmina"
            smile_txt = "Sorridente" if s == 1 else "Non sorridente"
            young_txt = "Giovane" if y == 1 else "Non giovane"
            label_text = f"{gender_txt}\n{smile_txt}\n{young_txt}\nclass={cls}"

            ax_text = axes[row_idx, 0]
            ax_text.text(
                0.5, 0.5, label_text,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                transform=ax_text.transAxes
            )
            ax_text.axis("off")

            for j in range(batch_size):
                ax = axes[row_idx, j + 1]
                img = gen[j].cpu().permute(1, 2, 0)
                ax.imshow(img)
                ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(
        OUTPUT_DIR,
        f"ddpm_generation_grid_{batch_size}samples_epoch_{checkpoint['epoch']}.png"
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved DDPM generation grid to {out_path}", flush=True)


generate_condition_combinations_and_save(
    model,
    checkpoint,
    batch_size=BATCH_SIZE,
    lam=LAM
)