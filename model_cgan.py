import io
import os
import base64
import random
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

KEYWORD_MAP = {
    0: ['airplane','plane','jet','aircraft','fly','flight','aeroplane','aviation'],
    1: ['car','automobile','vehicle','sedan','coupe','racing','road','drive'],
    2: ['bird','eagle','hawk','parrot','robin','sparrow','owl','pigeon','crow','swan'],
    3: ['cat','kitten','feline','kitty','tabby','paw'],
    4: ['deer','tree','forest','nature','woods','antler','buck','doe','wildlife',
        'plant','leaf','leaves','park','meadow','jungle','bush','grove'],
    5: ['dog','puppy','canine','hound','labrador','golden','retriever','poodle'],
    6: ['frog','toad','amphibian','pond'],
    7: ['horse','stallion','mare','pony','equine','gallop'],
    8: ['ship','boat','vessel','ocean','sea','sailing','navy','harbor','harbour',
        'water','lake','river','shore','beach'],
    9: ['truck','lorry','cargo','semi','trailer','freight'],
}


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


def prompt_to_class(prompt: str) -> int:
    p = prompt.lower()
    for cls_idx, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in p:
                return cls_idx
    return random.randint(0, 9)


def get_class_name(prompt: str) -> str:
    return CIFAR10_CLASSES[prompt_to_class(prompt)]


_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generator_v2.pth")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_gen_v2 = None


def _load_v2():
    global _gen_v2
    if _gen_v2 is None and os.path.exists(_CKPT):
        try:
            m = ConditionalGenerator().to(_device)
            m.load_state_dict(torch.load(_CKPT, map_location=_device))
            m.eval()
            _gen_v2 = m
        except Exception:
            pass  # old FC checkpoint incompatible with DCGAN — falls back to v1
    return _gen_v2


def make_image_v2(prompt: str = "") -> str:
    """Generate a class-conditioned image. Falls back to v1 MNIST model if not trained yet."""
    gen = _load_v2()
    if gen is None:
        from model import make_image
        return make_image(prompt)

    cls_idx = prompt_to_class(prompt)
    with torch.no_grad():
        z = torch.randn(1, 100, device=_device)
        labels = torch.tensor([cls_idx], device=_device)
        out = gen(z, labels)

    arr = out.squeeze().permute(1, 2, 0).cpu().numpy()
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr, mode="RGB")
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=2))
    img = img.resize((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
