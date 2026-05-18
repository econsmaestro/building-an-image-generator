import io
import json
import base64
import requests
from flask import Blueprint,jsonify,request
from model import make_image
from PIL import Image,ImageEnhance,ImageFilter

chat_bp=Blueprint("chat",__name__)

OLLAMA_URL="http://localhost:11434/api/chat"
MODEL="llama3.2"

# last generated image kept in memory for the enhance tool
_state={"last_image":None,"last_count":0}

SYSTEM="""You are an assistant for an AI image generator powered by Stable Diffusion.
You can generate any image the user describes — animals, landscapes, portraits, abstract art, anything.
Always call generate_image with a detailed prompt when the user asks to create or generate an image.
Be creative: expand the user's description into a vivid, detailed prompt for better results.
Always use tools when generating or enhancing."""

TOOLS=[
    {
        "type":"function",
        "function":{
            "name":"generate_image",
            "description":"Generate any image from a text description using Stable Diffusion",
            "parameters":{
                "type":"object",
                "properties":{
                    "prompt":{"type":"string","description":"Detailed description of the image to generate"}
                },
                "required":["prompt"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"enhance_image",
            "description":"Enhance the last generated image with visual adjustments",
            "parameters":{
                "type":"object",
                "properties":{
                    "brightness":{"type":"number","description":"Brightness multiplier, 1.0 is unchanged"},
                    "contrast":{"type":"number","description":"Contrast multiplier, 1.0 is unchanged"},
                    "sharpness":{"type":"number","description":"Sharpness multiplier, 1.0 is unchanged"},
                    "blur":{"type":"number","description":"Gaussian blur radius, 0 means no blur"}
                },
                "required":[]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"describe_last_image",
            "description":"Get information about the last generated image",
            "parameters":{"type":"object","properties":{},"required":[]}
        }
    }
]

def run_tool(name,args):
    """Execute a tool and return (result_text, image_b64 or None)."""
    if name=="generate_image":
        prompt=args.get("prompt","a beautiful painting")
        img=make_image(prompt)
        _state["last_image"]=img
        _state["last_count"]=1
        return f"Generated image: {prompt}",img

    elif name=="enhance_image":
        if not _state["last_image"]:
            return "No image generated yet — generate images first.",None
        brightness=float(args.get("brightness",1.0))
        contrast=float(args.get("contrast",1.0))
        sharpness=float(args.get("sharpness",1.0))
        blur=float(args.get("blur",0))
        img=Image.open(io.BytesIO(base64.b64decode(_state["last_image"]))).convert("L")
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
        enhanced=base64.b64encode(buf.getvalue()).decode("utf-8")
        _state["last_image"]=enhanced
        return f"Image enhanced (brightness={brightness}, contrast={contrast}, sharpness={sharpness}, blur={blur}).",enhanced

    elif name=="describe_last_image":
        if not _state["last_image"]:
            return "No image has been generated yet.",None
        return "An image was generated using Stable Diffusion based on the last prompt.",None

    return "Unknown tool.",None

@chat_bp.route("/chat",methods=["POST"])
def chat():
    data=request.get_json()
    message=data.get("message","")
    history=data.get("history",[])

    messages=[{"role":"system","content":SYSTEM}]+history+[{"role":"user","content":message}]
    image_b64=None
    reply_text=""

    # agentic loop — keep going until model stops calling tools
    for _ in range(6):
        try:
            resp=requests.post(OLLAMA_URL,json={
                "model":MODEL,
                "messages":messages,
                "tools":TOOLS,
                "stream":False
            },timeout=60).json()
        except Exception as e:
            return jsonify({"reply":f"Ollama error: {e}","image":None,"history":history})

        msg=resp.get("message",{})
        tool_calls=msg.get("tool_calls",[])

        if not tool_calls:
            reply_text=msg.get("content","")
            messages.append({"role":"assistant","content":reply_text})
            break

        # append assistant tool-call message
        messages.append(msg)

        # execute each tool and feed results back
        for tc in tool_calls:
            fn=tc.get("function",{})
            name=fn.get("name","")
            args=fn.get("arguments",{})
            if isinstance(args,str):
                args=json.loads(args)
            result_text,result_img=run_tool(name,args)
            if result_img:
                image_b64=result_img
            messages.append({"role":"tool","content":result_text})

    # strip system message and keep last 20 turns for history
    new_history=[m for m in messages if m.get("role")!="system"][-20:]

    return jsonify({
        "reply":reply_text or "Done!",
        "image":image_b64,
        "history":new_history
    })
