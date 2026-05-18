import io
import os
import base64
import threading
from flask import Flask, jsonify

app = Flask(__name__)

generator = None
model_ready = False

def load_model():
    global generator, model_ready
    import torch
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.fc1 = nn.Linear(100, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, 1024)
            self.fc4 = nn.Linear(1024, 28 * 28 * 1)

        def forward(self, z):
            x = torch.relu(self.fc1(z))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            return x.view(-1, 1, 28, 28)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = Generator().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "generator.pth")
    g.load_state_dict(torch.load(model_path, map_location=device))
    g.eval()
    generator = g
    model_ready = True
    print("Model loaded and ready.")

threading.Thread(target=load_model, daemon=True).start()

@app.route("/generate")
def generate():
    if not model_ready:
        return jsonify({"error": "Model is still loading, please try again in a moment."}), 503

    import torch
    from torchvision.utils import make_grid
    from PIL import Image

    device = next(generator.parameters()).device
    z = torch.randn(16, 100).to(device)
    with torch.no_grad():
        images = generator(z)
    images = (images + 1) / 2
    grid = make_grid(images, nrow=4, padding=2)
    img_array = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(img_array if img_array.ndim == 3 else img_array[:, :, 0])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return jsonify({"image": b64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))
    app.run(host="0.0.0.0", port=port, debug=False)
