from flask import Blueprint, jsonify, request
from model_cgan import make_image_v2, get_class_name

generate_bp = Blueprint("generate", __name__)

@generate_bp.route("/generate")
def generate():
    prompt = request.args.get("prompt", "a beautiful painting")
    try:
        image = make_image_v2(prompt)
        cls = get_class_name(prompt)
        return jsonify({"image": image, "class": cls})
    except Exception as e:
        return jsonify({"error": str(e), "image": None}), 500
