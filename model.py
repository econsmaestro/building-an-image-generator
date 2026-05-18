import io
import os
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x.view(-1, 1, 28, 28)

device = "cuda" if torch.cuda.is_available() else "cpu"
_generator = Generator().to(device)
_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generator.pth")
_generator.load_state_dict(torch.load(_ckpt, map_location=device))
_generator.eval()

def make_image(prompt="", steps=25):
    with torch.no_grad():
        z = torch.randn(1, 100, device=device)
        out = _generator(z)
    arr = out.squeeze().cpu().numpy()
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr, mode="L").resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
