import os
import json
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy

from torchvision.datasets import CelebA
from torchvision.utils import save_image, make_grid

# Immagini 64x64 
IMAGE_SIZE = 64

# ---- Transforms: Resize + CenterCrop + ToTensor + Normalize -> [-1,1] ----
try:
    from torchvision.transforms import v2 as T  # torchvision >= 0.15
    def build_transform():
        return T.Compose([
            T.ToImage(),
            # resize e crop per mantenere aspect ratio
            # facendo resize (imgsz. imgsz) si rischia
            # di deformare l'immagine se non è quadrata
            T.Resize(IMAGE_SIZE),          # mantiene aspect ratio (lato corto -> image_size)
            T.CenterCrop(IMAGE_SIZE),      # crop centrale quadrato
            T.ToDtype(torch.float32, scale=True),  # converte in float 32 e scala i pixel nel range [0,1]
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # normalizzazione [-1,1] per corenza di scala con tanh
        ])
except Exception: # fallback per torchvision < 0.15
    from torchvision import transforms as T
    def build_transform():
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),  # [0,1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ])


def attrs_to_class(attr: torch.Tensor) -> torch.Tensor:
    """
    attr: (B, 40) con valori -1/+1
    usa attributi: #20 male, #31 smiling, #39 young
    mappa in 8 classi: cls = 4*male + 2*smiling + 1*young
    """
    male = (attr[:, 20] == 1).long() # long converte in int[0,1]
    smiling = (attr[:, 31] == 1).long()
    young = (attr[:, 39] == 1).long()
    return male * 4 + smiling * 2 + young # codifica a 3 bit
    
    # Esempio:
    # male = 1, smiling = 1, young = 1
    # cls = 4*male + 2*smiling + 1*young = 4*1 + 2*1 + 1*1 = 7

def weights_init_normal(m):
    '''
    Inizializza pesi di conv e batchnorm con distribuzioni normali 
    come da DCGAN paper. Di default PyTorch usa Kaiming Uniform, che
    ha una varianza maggiore e può portare a instabilità nel training:
    può succedere che il discriminator impari troppo velocemente rispetto al
    generator, portando a mode collapse o a mancata convergenza.
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_size: int = 128, classes: int = 8, image_channels: int = 3):
        super().__init__()
        self.latent_size = latent_size
        self.classes = classes
        self.image_channels = image_channels

        self.e = None  # one-hot eye 

        # Fully connected: (latent+classes) -> 512*4*4 linear output: (B, latent+classes) -> (B, 8192)
        # la batchnorm cancella il bias del layer precedente
        # e lo sostituisce con il beta appreso, quindi 
        # si può mettere bias=False per il layer precedente
        # in modo da risparmiare memoria e calcoli
        self.fc = nn.Linear(latent_size + classes, 512 * 4 * 4, bias = False)

        # Upsampling: 4x4 -> 8 -> 16 -> 32 -> 64
        self.net = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)), # unflatten della dimensione 1: (B,8192) -> (B,512,4,4) -> Hin = 4

            nn.BatchNorm2d(512), # batchnorm 2D per immagini uguale al numero di canali in input
            nn.ReLU(True),

            # espandiamo la dimensione spaziale con ConvTranspose2d, riducendo i canali (profondità)
            # Hout = (Hin-1)*stride - 2*padding + kernel_size + output_padding
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias = False),  # 8x8
            nn.BatchNorm2d(256), # batchnorm 2D per immagini uguale al numero di canali in output dal layer precedente
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias = False),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias = False),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias = False),    # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, image_channels, 3, padding=1),
            # no batchnorm all'ultimo layer (output finale)
            nn.Tanh(),  # output in [-1,1] 
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.e is None or self.e.device != z.device:
            self.e = torch.eye(self.classes, device=z.device)
        c = self.e[c]  # one-hot (B,classes)
        zc = torch.cat((z, c), dim=1)  # (B, latent+classes)
        x = self.fc(zc)               # (B, 512*4*4)
        return self.net(x)            # (B,3,64,64) in [-1,1]


class Discriminator(nn.Module):
    def __init__(self, classes: int = 8, image_channels: int = 3):
        super().__init__()
        self.classes = classes
        self.image_channels = image_channels

        self.e = None  # one-hot eye 

        # 64x64 -> 32x32 -> 16x16
        # net1 estrae feature dall'immagine
        self.net1 = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, stride=2, padding=1),   # 32x32
            nn.LeakyReLU(0.2, inplace=True), # funzione di attivazione LeakyReLU
                                             # anche qui può essere inplace=True per risparmio memoria
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),              # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True), 
        )

        # concat condizione su 16x16 -> 8x8 -> 4x4 -> score
        self.net2 = nn.Sequential(
            nn.Conv2d(128 + classes, 256, 4, stride=2, padding=1, bias=False),   # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),             # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.075), # dropout per regolarizzazione, non troppo alto per evitare underfitting
                             # essendo che utilizziamo anche la label smoothing e instance noise
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.e is None or self.e.device != x.device:
            self.e = torch.eye(self.classes, device=x.device)
        c = self.e[c]  # (B,classes)

        y1 = self.net1(x)  # (B,128,16,16)
        c_map = c[:, :, None, None].expand(-1, -1, y1.shape[2], y1.shape[3])  # (B,classes,16,16)
                # agiungo due dimensioni spaziali vuote per arrivare a 4D e le espando a 16x16
        yc = torch.cat((y1, c_map), dim=1)  # concatenazione sui canali [dim=1] (B,128+classes,16,16)
        return self.net2(yc)  # (B,1)


@torch.no_grad()
def save_samples(
    gen: Generator,
    out_png: Path,
    device: torch.device,
    latent_size: int,
    classes: int,
    n_per_class: int,
    seed: int 
):
    """Salva una griglia con 8 colonne (una per classe). Output gen in [-1,1] -> salvataggio normalizzato."""
    gen.eval()

    g = None
    
    g = torch.Generator(device=str(device))
    g.manual_seed(seed)

    n = classes * n_per_class
    cls = torch.arange(classes, device=device).repeat_interleave(n_per_class)  # (n,)
    z = torch.randn(n, latent_size, device=device, generator=g)
    x = gen(z, cls).cpu()  # (n,3,64,64) in [-1,1]

    grid = make_grid(x, nrow=classes)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # normalize=True rimappa da [-1,1] a [0,1] per il salvataggio
    save_image(grid, str(out_png), normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/pfoggia/GenerativeAI/CELEBA")
    parser.add_argument("--download", action="store_true", help="comando per scaricare CelebA se non presente")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="runs/cgan_celeba")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--sample_n_per_class", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5, help="Salva ckpt_epoch_XXX ogni N epoche")
    parser.add_argument("--max_batches", type=int, default=0, help="Debug: limita i batch per epoca (0=nessun limite)")

    # Device: usa TORCH_DEVICE se presente, altrimenti auto
    default_device = os.getenv("TORCH_DEVICE", "")
    parser.add_argument(
        "--device",
        type=str,
        default=(default_device if default_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    )

    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA non disponibile: passo a CPU")
        device = torch.device("cpu")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints" # Cartella per i checkpoint
    sample_dir = out_dir / "samples"
    sample_dir_fixed = sample_dir / "fixed"    # Cartella per seed 123
    sample_dir_random = sample_dir / "random"  # Cartella per seed variabili

    # Crea cartelle di output
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir_fixed.mkdir(parents=True, exist_ok=True)
    sample_dir_random.mkdir(parents=True, exist_ok=True)

    # Salva config utile anche per inferenza
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "device": str(device)}, f, indent=2)

    transform = build_transform()

    dataset = CelebA(
        root=args.data_root,
        split="all",
        target_type="attr",
        transform=transform,
        download=args.download
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True # cancella l'ultimo batch se incompleto, in modo da avere sempre batch completi
    )

    classes = 8  # 3 bit => 8 combinazioni
    gen = Generator(latent_size=args.latent_size, classes=classes, image_channels=3).to(device)
    disc = Discriminator(classes=classes, image_channels=3).to(device)

    gen.apply(weights_init_normal)
    disc.apply(weights_init_normal)

    # Ottimizzatori, momentum può portare instabilità in GAN
    # quindi betas=(0.5, 0.999) per ridurre momentum
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        gen.load_state_dict(ckpt["gen_state_dict"])
        disc.load_state_dict(ckpt["disc_state_dict"])
        gen_opt.load_state_dict(ckpt["gen_opt_state_dict"])
        disc_opt.load_state_dict(ckpt["disc_opt_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[resume] Riparto da epoch {start_epoch}")

    def disc_loss_function(d_true, d_synth):
        # label smoothing per confondere il discriminator
        # cercando di evitare che impari a distinguere troppo facilmente
        # i veri dai sintetici, annullando le capacità di apprendimento del generator
        t_true = torch.ones_like(d_true) - args.label_smoothing   # 0.9
        t_synth = torch.zeros_like(d_synth) + args.label_smoothing # 0.1
        return binary_cross_entropy(d_true, t_true) + binary_cross_entropy(d_synth, t_synth)

    def gen_loss_function(d_synth):
        t_synth = torch.ones_like(d_synth)
        return binary_cross_entropy(d_synth, t_synth)
    
    # File di log per visualizzare andamento training
    log_file = out_dir / "history.csv"
    with open(log_file, "w") as f:
        f.write("epoch,gloss,dloss,dtrue,dsynth\n")

    # instance noise per stabilizzare il training, evitando che D impari troppo velocemente
    # quindi passiamo a D immagini leggermente rumorose
    start_noise = 0.07
    end_noise = 0.00

    for epoch in range(start_epoch, args.epochs):
        
        # per visualizzare e salvare, le epoche partono da 1
        epoch_display = epoch + 1

        gen.train()
        disc.train()

        sum_gloss = 0.0
        sum_dloss = 0.0
        sum_dtrue = 0.0
        sum_dsynth = 0.0
        batches = 0

        # Lineare decay del rumore, diminuisce con il passare delle epoche
        noise_std = start_noise - (start_noise - end_noise) * (epoch/args.epochs)
        noise_std = max(end_noise, noise_std)

        for i, (x_true, attr) in enumerate(loader):

            if args.max_batches and i >= args.max_batches:
                break

            x_true = x_true.to(device)
            attr = attr.to(device)
            cls = attrs_to_class(attr)

            # --------- Discriminator step ---------
            z = torch.randn(x_true.shape[0], args.latent_size, device=device)
            x_synth = gen(z, cls)

            # randn_like per avere un tensore della stessa forma di x_true
            if noise_std > 0:
                x_true_d  = x_true + torch.randn_like(x_true) * noise_std
                # detach: non aggiorno i gradienti per G nel passo di D
                # qui il generator mi serve solo per produrre x_synth
                x_synth_d = x_synth.detach() + torch.randn_like(x_synth) * noise_std  
            else:
                x_true_d  = x_true
                x_synth_d = x_synth.detach()

            # aggiungiamo rumore (diverso) sia per i veri che per i sintetici
            d_true = disc(x_true_d, cls)
            d_synth = disc(x_synth_d, cls)  

            disc_opt.zero_grad()
            dloss = disc_loss_function(d_true, d_synth)
            dloss.backward()
            disc_opt.step()
            # --------- Generator step ---------
            d_synth_for_g = disc(x_synth, cls)
            gen_opt.zero_grad()
            gloss = gen_loss_function(d_synth_for_g)
            gloss.backward()
            gen_opt.step()

            sum_gloss += float(gloss.detach().cpu())
            sum_dloss += float(dloss.detach().cpu())
            sum_dtrue += float(d_true.mean().detach().cpu())
            sum_dsynth += float(d_synth_for_g.mean().detach().cpu())
            batches += 1

        avg_gloss = sum_gloss / max(batches, 1)
        avg_dloss = sum_dloss / max(batches, 1)
        avg_dtrue = sum_dtrue / max(batches, 1)
        avg_dsynth = sum_dsynth / max(batches, 1)

        print(
            f"[epoch {epoch_display}] "
            f"GLoss={avg_gloss:.4f} "
            f"DLoss={avg_dloss:.4f} "
            f"Dtrue={avg_dtrue:.4f} "
            f"Dsynth={avg_dsynth:.4f}"
        )

        with open(log_file, "a") as f:
            f.write(f"{epoch_display},{avg_gloss:.6f},{avg_dloss:.6f},{avg_dtrue:.6f},{avg_dsynth:.6f}\n")

        # ----------------- checkpoint policy -----------------
        ckpt = {
            "epoch": epoch_display,
            "gen_state_dict": gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "gen_opt_state_dict": gen_opt.state_dict(),
            "disc_opt_state_dict": disc_opt.state_dict(),
            "latent_size": args.latent_size,
            "classes": classes,
            "image_size": IMAGE_SIZE,
        }

        # latest: SEMPRE (si sovrascrive)
        torch.save(ckpt, ckpt_dir / "latest.pt")

        # ckpt ogni N epoche
        if (epoch_display % args.save_every) == 0:
            torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch_display:03d}.pt")

        # ----------------- samples -----------------
        
        # samples ogni N epoche, salvando sia con seed fisso che variabile
        # in modo da valutare sia la qualità che la variabilità del modello
        # durante il training

        if (epoch_display % args.sample_every) == 0:
            out_png_fixed = sample_dir_fixed / f"samples_FIXED_epoch_{epoch_display:03d}.png"
            save_samples(
                gen=gen,
                out_png=out_png_fixed,
                device=device,
                latent_size=args.latent_size,
                classes=classes,
                n_per_class=args.sample_n_per_class,
                seed=123  # <-- seed fisso
            )


            out_png = sample_dir_random / f"samples_RANDOM_epoch_{epoch_display:03d}.png"
            save_samples(
                gen=gen,
                out_png=out_png,
                device=device,
                latent_size=args.latent_size,
                classes=classes,
                n_per_class=args.sample_n_per_class,
                seed=epoch_display # <-- seed variabile
            )

    print(f"Finito. Checkpoints in: {ckpt_dir}")
    print(f"Samples in: {sample_dir}")


if __name__ == "__main__":
    main()
