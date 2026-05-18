from flask import Blueprint, jsonify, request
from model import make_image

generate_bp = Blueprint("generate", __name__)

@generate_bp.route("/generate")
def generate():
    prompt = request.args.get("prompt", "a beautiful painting")
    try:
        return jsonify({"image": make_image(prompt)})
    except Exception as e:
        return jsonify({"error": str(e), "image": None}), 500
