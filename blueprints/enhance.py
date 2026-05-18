import io
import base64
from flask import Blueprint,jsonify,request
from PIL import Image,ImageEnhance,ImageFilter

enhance_bp=Blueprint("enhance",__name__)

@enhance_bp.route("/enhance",methods=["POST"])
def enhance():
    file=request.files.get("image")
    brightness=float(request.form.get("brightness",1.0))
    contrast=float(request.form.get("contrast",1.0))
    sharpness=float(request.form.get("sharpness",1.0))
    blur=float(request.form.get("blur",0))
    img=Image.open(file).convert("L")
    if brightness!=1.0:
        img=ImageEnhance.Brightness(img).enhance(brightness)
    if contrast!=1.0:
        img=ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness!=1.0:
        img=ImageEnhance.Sharpness(img).enhance(sharpness)
    if blur>0:
        img=img.filter(ImageFilter.GaussianBlur(radius=blur))
    buf=io.BytesIO()
    img.save(buf,format="PNG")
    return jsonify({"image":base64.b64encode(buf.getvalue()).decode("utf-8")})
