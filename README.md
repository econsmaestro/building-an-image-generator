# Building an Image Generator

A full-stack image generation app that combines a trained GAN model with modern AI image generation.

## Features

- **Text to Image** — Describe anything and generate it for free using Pollinations AI (FLUX model)
- **Enhance Image** — Upload a photo and describe changes using InstructPix2Pix (e.g. "make it look like winter")
- **My GAN Model** — Generates handwritten digits using a trained GAN on MNIST

## Project Structure

```
app.py                    # Original Flask app (GAN only)
templates/index.html      # Original HTML template
gan_backend/app.py        # Enhanced Flask backend (serves the GAN model)
api_server/               # Node.js/Express API server
  src/app.ts              # Express app setup
  src/routes/generate.ts  # Routes: GAN, text-to-image, image enhancement
  src/lib/python_backend.ts # Spawns the Python GAN backend
frontend/                 # React + Vite frontend
  src/App.tsx             # Main app with 3 tabs
  src/index.css           # Dark theme styling
  src/main.tsx            # Entry point
```

## Setup

### Original Flask App (GAN only)
```bash
pip install flask torch torchvision pillow
python app.py
```
> Requires `generator.pth` in the same directory (trained MNIST GAN model)

### Full Stack App
```bash
pnpm install
pnpm --filter @workspace/api-server run dev
pnpm --filter @workspace/gan-image-generator run dev
```

## How It Works

1. **GAN Backend** — A Flask server loads `generator.pth` and exposes a `/generate` endpoint that runs inference and returns a 4x4 grid of generated digit images as base64 PNG
2. **API Server** — An Express server spawns the Flask backend as a child process and proxies requests to it. Also handles text-to-image (Pollinations AI) and image enhancement (HuggingFace InstructPix2Pix)
3. **Frontend** — A React app with three tabs, dark themed, matching the original design

## Model

The GAN model (`generator.pth`) is a simple fully-connected generator trained on MNIST:
- Input: 100-dimensional random noise vector
- Output: 28×28 grayscale images (16 at a time arranged in a 4×4 grid)
- Architecture: 4 linear layers (100→256→512→1024→784) with ReLU activations and tanh output
