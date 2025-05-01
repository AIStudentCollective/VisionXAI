from flask import Flask, request, redirect
from flask_cors import CORS
from tensorflow_route import tensorflow_bp, handle_tensorflow_request
from pytorch import pytorch_bp

app = Flask(__name__)
# Configure CORS with specific settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "https://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

@app.route("/")
def home():
    return "Flask server is running!"

# Add a compatibility route to match the frontend's expectations
# @app.route("/process-image", methods=["POST"])
# def process_image_compat():
#     # This route redirects requests to the appropriate endpoint based on the architecture
#     architecture = request.form.get('architecture', '')
    
#     # For now, redirect all requests to the TensorFlow endpoint
#     # You can add more logic here if needed
#     return tensorflow_bp.handle_tensorflow_request()

app.register_blueprint(pytorch_bp, url_prefix='/api/pytorch')
app.register_blueprint(tensorflow_bp, url_prefix='/api/tensorflow')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
