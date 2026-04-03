import os
import argparse
import torch
from pathlib import Path
from torchvision.utils import save_image, make_grid

from train_cgan import Generator

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Percorso del file .pt")
    parser.add_argument("--out_dir", type=str, default="results", help="Cartella di output")
    parser.add_argument("--n_per_class", type=int, default=8, help="Quante immagini generare per ogni classe (colonne)")
    parser.add_argument("--seed", type=int, default=123, help="Seed per la riproducibilità")
    
    default_device = os.getenv("TORCH_DEVICE", "")
    parser.add_argument(
        "--device",
        type=str,
        default=(default_device if default_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    )

    args = parser.parse_args()

    # Setup Device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA non disponibile: passo a CPU")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Caricamento Checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Recupero parametri salvati nel checkpoint
    latent_size = ckpt.get("latent_size", 128)
    classes = ckpt.get("classes", 8)
    
    # Inizializzazione Generator
    gen = Generator(latent_size=latent_size, classes=classes, image_channels=3).to(device)
    gen.load_state_dict(ckpt["gen_state_dict"])
    gen.eval()

    # Setup Generatore locale per il seed
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    
    print(f"Generating samples with seed: {args.seed}")

    # Creazione input:
    # Generiamo n_per_class immagini per ogni classe (0..7)
    # repeat_interleave fa: [0,0,0, 1,1,1, ...]
    n = classes * args.n_per_class
    cls_labels = torch.arange(classes, device=device).repeat_interleave(args.n_per_class)
    
    z = torch.randn(n, latent_size, device=device, generator=g)

    # Inferenza
    images = gen(z, cls_labels)

    # Salvataggio Griglia
    # nrow=args.n_per_class assicura che ogni RIGA corrisponda a una CLASSE
    grid_path = out_dir / f"generated_seed_{args.seed}.png"
    
    grid = make_grid(images, nrow=args.n_per_class, padding=2, normalize=True, value_range=(-1, 1))
    save_image(grid, str(grid_path))

    print(f"Fatto! Griglia salvata in: {grid_path}")

if __name__ == "__main__":
    main()