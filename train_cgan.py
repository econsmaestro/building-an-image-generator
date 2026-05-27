"""
Train a Conditional GAN on CIFAR-10.
Saves the generator to generator_v2.pth in the same directory.

Usage:
    python train_cgan.py                  # 50 epochs, default settings
    python train_cgan.py --epochs 100     # more epochs = better quality
    python train_cgan.py --epochs 10      # quick smoke-test
"""

import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model_cgan import ConditionalGenerator

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 64 * 64)
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),            # (4,64,64) → (64,32,32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),           # → (128,16,16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),          # → (256,8,8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),          # → (512,4,4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_map = self.embed(labels).view(-1, 1, 64, 64)
        return self.net(torch.cat([img, label_map], dim=1))


def train(epochs=50, batch_size=128, lr=2e-4, save_every=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training for {epochs} epochs  |  batch={batch_size}  |  lr={lr}")
    print("Downloading CIFAR-10 if needed...")

    transform = T.Compose([T.Resize(64), T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    G = ConditionalGenerator().to(device)
    D = ConditionalDiscriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generator_v2.pth")

    for epoch in range(1, epochs + 1):
        g_total = d_total = 0.0
        for real_imgs, labels in loader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            bs = real_imgs.size(0)
            real_t = torch.ones(bs, 1, device=device)
            fake_t = torch.zeros(bs, 1, device=device)

            # Discriminator step
            z = torch.randn(bs, 100, device=device)
            fake_imgs = G(z, labels).detach()
            d_loss = criterion(D(real_imgs, labels), real_t) + criterion(D(fake_imgs, labels), fake_t)
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # Generator step
            z = torch.randn(bs, 100, device=device)
            fake_imgs = G(z, labels)
            g_loss = criterion(D(fake_imgs, labels), real_t)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            g_total += g_loss.item()
            d_total += d_loss.item()

        n = len(loader)
        print(f"Epoch {epoch:>3}/{epochs}  G_loss={g_total/n:.4f}  D_loss={d_total/n:.4f}")

        if epoch % save_every == 0 or epoch == epochs:
            torch.save(G.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved to {ckpt_path}")

    print(f"\nTraining complete. Model saved to {ckpt_path}")
    print("Restart app.py to load the new model.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_every=args.save_every)
