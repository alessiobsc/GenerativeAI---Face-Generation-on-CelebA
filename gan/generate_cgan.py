import os
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from train_cgan_G1024 import Generator


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Percorso del file .pt")
    parser.add_argument("--out_dir",    type=str, default="results", help="Cartella di output")
    parser.add_argument("--n_per_class", type=int, default=5, help="Quante immagini per ogni classe (colonne)")
    parser.add_argument("--seed",       type=int, default=123, help="Seed per la riproducibilità")

    default_device = os.getenv("TORCH_DEVICE", "")
    parser.add_argument(
        "--device",
        type=str,
        default=(default_device if default_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA non disponibile: passo a CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Caricamento checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    latent_size = ckpt.get("latent_size", 128)
    classes     = ckpt.get("classes", 8)

    gen = Generator(latent_size=latent_size, classes=classes, image_channels=3).to(device)
    gen.load_state_dict(ckpt["gen_state_dict"])
    gen.eval()

    # ---- Seed riproducibile ----
    g = torch.Generator(device=str(device))
    g.manual_seed(args.seed)

    # ---- Generazione di tutti i campioni insieme ----
    n = classes * args.n_per_class
    cls = torch.arange(classes, device=device).repeat_interleave(args.n_per_class)  # (n,)
    z = torch.randn(n, latent_size, device=device, generator=g)
    images = gen(z, cls)  # (n, 3, 64, 64) in [-1,1]

    # Da [-1,1] a [0,1] per matplotlib
    images = (images.clamp(-1, 1) + 1) / 2

    # ---- Tutte le 8 combinazioni di condizioni ----
    all_conditions = [
        (m, s, y)
        for m in [0, 1]
        for s in [0, 1]
        for y in [0, 1]
    ]

    num_rows      = len(all_conditions)          # 8
    num_cols_total = 1 + args.n_per_class        # etichetta + immagini

    fig, axes = plt.subplots(
        num_rows,
        num_cols_total,
        figsize=(2.6 * num_cols_total, 2.3 * num_rows)
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for row_idx, (m, s, y) in enumerate(all_conditions):
        cls_idx = m * 4 + s * 2 + y  # indice classe 0..7

        # ---- Colonna 0: etichetta testuale ----
        gender_txt = "Maschio"       if m == 1 else "Femmina"
        smile_txt  = "Sorridente"    if s == 1 else "Non sorridente"
        young_txt  = "Giovane"       if y == 1 else "Non giovane"
        label_text = f"{gender_txt}\n{smile_txt}\n{young_txt}\nclass={cls_idx}"

        ax_text = axes[row_idx, 0]
        ax_text.text(
            0.5, 0.5, label_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            transform=ax_text.transAxes
        )
        ax_text.axis("off")

        # ---- Colonne 1..n: immagini generate ----
        for j in range(args.n_per_class):
            idx = row_idx * args.n_per_class + j
            ax = axes[row_idx, j + 1]
            img = images[idx].cpu().permute(1, 2, 0)  # (H, W, 3)
            ax.imshow(img)
            ax.axis("off")

    plt.suptitle(f"CGAN CelebA — seed={args.seed} — epoch={ckpt.get('epoch', '?')}", fontsize=13)
    plt.tight_layout()

    out_path = out_dir / f"cgan_generation_grid_{args.n_per_class}samples_epoch_{ckpt.get('epoch', 'X')}.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Griglia salvata in: {out_path}")


if __name__ == "__main__":
    main()