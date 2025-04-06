from flask import Flask
from pytorch import pytorch_bp

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask server is running!"

app.register_blueprint(pytorch_bp, url_prefix='/api/pytorch')

if __name__ == "__main__":
    app.run(debug=True)
