import io
import base64
import torch
import torch.nn as nn
from flask import Flask,render_template,jsonify
from torchvision.utils import make_grid

app=Flask(__name__)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1=nn.Linear(100,256)
        self.fc2=nn.Linear(256,512)
        self.fc3=nn.Linear(512,1024)
        self.fc4=nn.Linear(1024,28*28*1)
    def forward(self,z):
        x=torch.relu(self.fc1(z))
        x=torch.relu(self.fc2(x))
        x=torch.relu(self.fc3(x))
        x=torch.tanh(self.fc4(x))
        return x.view(-1,1,28,28)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator=Generator().to(device)
generator.load_state_dict(torch.load("generator.pth",map_location=device))
generator.eval()

def generate_image():
    z=torch.randn(16,100).to(device)
    with torch.no_grad():
        images=generator(z)
    images=(images+1)/2 #denormalize from [-1,1] to [0,1]
    grid=make_grid(images,nrow=4,padding=2)
    img_array=(grid.permute(1,2,0).cpu().numpy()*255).astype("uint8")
    from PIL import Image
    img=Image.fromarray(img_array.squeeze(),"L")
    buf=io.BytesIO()
    img.save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate")
def generate():
    img_b64=generate_image()
    return jsonify({"image":img_b64})

if __name__=="__main__":
    app.run(debug=True)
