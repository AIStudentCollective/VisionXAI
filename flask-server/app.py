from flask import Flask
from tensorflow import tensorflow_bp
from pytorch import pytorch_bp

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask server is running!"

app.register_blueprint(pytorch_bp, url_prefix='/api/pytorch')
app.register_blueprint(tensorflow_bp, url_prefix='/api/tensorflow')

if __name__ == "__main__":
    app.run(debug=True)
