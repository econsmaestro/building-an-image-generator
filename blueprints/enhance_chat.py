import io
import json
import base64
import requests
from flask import Blueprint, jsonify, request
from PIL import Image, ImageEnhance, ImageFilter

enhance_chat_bp = Blueprint("enhance_chat", __name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

SYSTEM = """You are a specialized image enhancement assistant. The user has uploaded an image and wants to improve it visually.
You can adjust brightness, contrast, sharpness, and blur. Always call enhance_image when the user describes any visual change they want.
Interpret natural language into sensible parameter values. Be concise — just confirm what you applied."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "enhance_image",
            "description": "Apply visual enhancements to the uploaded image",
            "parameters": {
                "type": "object",
                "properties": {
                    "brightness": {"type": "number", "description": "Brightness multiplier 0.1-3.0, 1.0 = unchanged"},
                    "contrast":   {"type": "number", "description": "Contrast multiplier 0.1-3.0, 1.0 = unchanged"},
                    "sharpness":  {"type": "number", "description": "Sharpness multiplier 0-5.0, 1.0 = unchanged"},
                    "blur":       {"type": "number", "description": "Gaussian blur radius 0-5.0, 0 = no blur"}
                },
                "required": []
            }
        }
    }
]

def _apply(image_b64, brightness=1.0, contrast=1.0, sharpness=1.0, blur=0.0):
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("L")
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@enhance_chat_bp.route("/enhance-chat", methods=["POST"])
def enhance_chat():
    data      = request.get_json()
    message   = data.get("message", "")
    history   = data.get("history", [])
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"reply": "Please upload an image first.", "image": None, "history": history})

    messages    = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": message}]
    image_out   = None
    reply_text  = ""

    for _ in range(6):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL, "messages": messages, "tools": TOOLS, "stream": False
            }, timeout=60).json()
        except Exception as e:
            return jsonify({"reply": f"Ollama error: {e}", "image": None, "history": history})

        msg        = resp.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            reply_text = msg.get("content", "")
            messages.append({"role": "assistant", "content": reply_text})
            break

        messages.append(msg)

        for tc in tool_calls:
            fn   = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)

            brightness = float(args.get("brightness", 1.0))
            contrast   = float(args.get("contrast",   1.0))
            sharpness  = float(args.get("sharpness",  1.0))
            blur       = float(args.get("blur",        0.0))
            image_b64  = _apply(image_b64, brightness, contrast, sharpness, blur)
            image_out  = image_b64
            messages.append({"role": "tool", "content":
                f"Enhanced: brightness={brightness}, contrast={contrast}, sharpness={sharpness}, blur={blur}"})

    new_history = [m for m in messages if m.get("role") != "system"][-20:]
    return jsonify({"reply": reply_text or "Done!", "image": image_out, "history": new_history})
