"""
WGAN-GP Conditional GAN on CIFAR-10
Designed for Google Colab GPU (~30-40 min for 100 epochs on T4).

Steps:
  1. Upload this file to Colab (File > Upload)
  2. Runtime > Change runtime type > T4 GPU
  3. Run:  !python train_wgangp_colab.py
  4. Download generator_v2.pth when done, replace the one in your project folder
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.autograd as autograd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── Generator (identical to model_cgan.py — checkpoint is drop-in compatible) ──

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, embed_dim=10):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(noise_dim + embed_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 4x4 → 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 8x8 → 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),    # 16x16 → 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),     # 32x32 → 64x64
            nn.Tanh(),
        )

    def forward(self, z, labels):
        x = self.fc(torch.cat([z, self.embed(labels)], dim=1))
        return self.net(x.view(-1, 256, 4, 4))


# ── Critic (no BatchNorm, no Sigmoid — required for WGAN-GP) ──

class ConditionalCritic(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 64 * 64)
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),             # (4,64,64) → (64,32,32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),            # → (128,16,16)
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),           # → (256,8,8)
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),           # → (512,4,4)
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            # No Sigmoid — critic outputs raw score
        )

    def forward(self, img, labels):
        label_map = self.embed(labels).view(-1, 1, 64, 64)
        return self.net(torch.cat([img, label_map], dim=1))


def gradient_penalty(critic, real, fake, labels, device):
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(interpolated, labels)
    grad = autograd.grad(
        outputs=score, inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grad.view(bs, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def train(epochs=100, batch_size=256, lr=1e-4, n_critic=5, lambda_gp=10, save_every=10):
    print(f"Training WGAN-GP  |  epochs={epochs}  batch={batch_size}  n_critic={n_critic}")

    transform = T.Compose([T.Resize(64), T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    G = ConditionalGenerator().to(DEVICE)
    C = ConditionalCritic().to(DEVICE)

    # WGAN-GP uses Adam with betas=(0, 0.9)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))

    ckpt_path = "generator_v2.pth"
    full_ckpt_path = "checkpoint_wgangp.pth"
    start_epoch = 1

    if os.path.exists(full_ckpt_path):
        ckpt = torch.load(full_ckpt_path, map_location=DEVICE)
        G.load_state_dict(ckpt["G"])
        C.load_state_dict(ckpt["C"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_c.load_state_dict(ckpt["opt_c"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, epochs + 1):
        g_total = c_total = 0.0
        for real_imgs, labels in loader:
            real_imgs, labels = real_imgs.to(DEVICE), labels.to(DEVICE)
            bs = real_imgs.size(0)

            # ── Critic steps (n_critic per generator step) ──
            for _ in range(n_critic):
                z = torch.randn(bs, 100, device=DEVICE)
                fake_imgs = G(z, labels).detach()
                gp = gradient_penalty(C, real_imgs, fake_imgs, labels, DEVICE)
                c_loss = C(fake_imgs, labels).mean() - C(real_imgs, labels).mean() + lambda_gp * gp
                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            # ── Generator step ──
            z = torch.randn(bs, 100, device=DEVICE)
            fake_imgs = G(z, labels)
            g_loss = -C(fake_imgs, labels).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_total += g_loss.item()
            c_total += c_loss.item()

        n = len(loader)
        print(f"Epoch {epoch:>3}/{epochs}  G_loss={g_total/n:.4f}  C_loss={c_total/n:.4f}")

        if epoch % save_every == 0 or epoch == epochs:
            torch.save(G.state_dict(), ckpt_path)
            torch.save({"epoch": epoch, "G": G.state_dict(), "C": C.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_c": opt_c.state_dict()}, full_ckpt_path)
            print(f"  -> Saved {ckpt_path} + {full_ckpt_path}")

    print(f"\nDone. Download {ckpt_path} and replace the one in your project folder.")

    # Auto-download in Colab
    try:
        from google.colab import files
        files.download(ckpt_path)
        print("Download started.")
    except ImportError:
        print(f"Not in Colab — find {ckpt_path} in your working directory.")


if __name__ == "__main__":
    train()
