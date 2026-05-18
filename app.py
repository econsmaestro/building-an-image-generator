from flask import Flask,render_template
from blueprints.generate import generate_bp
from blueprints.chat import chat_bp
from blueprints.enhance import enhance_bp
from blueprints.enhance_chat import enhance_chat_bp

app=Flask(__name__)
app.register_blueprint(generate_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(enhance_bp)
app.register_blueprint(enhance_chat_bp)

@app.route("/")
def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)
