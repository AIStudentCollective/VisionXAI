from flask import Flask, request, jsonify, send_from_directory, Response, send_file, Blueprint
import os
import subprocess
import base64
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create a Blueprint for TensorFlow routes
tensorflow_bp = Blueprint('tensorflow', __name__)

@tensorflow_bp.route('/heatmap', methods=['POST'])
def tensorflow_heatmap():
    # Get uploaded files and architecture
    class_names_file = request.files.get('class_names')
    weights_file = request.files.get('weights')
    image_file = request.files.get('image')
    architecture = request.form.get('architecture')

    # Get the API directory (where app.py is located)
    api_dir = os.path.dirname(__file__)

    # File handling and validation
    try:
        uploads_dir = os.path.join(api_dir, 'uploads')  # Uploads directory within the API directory
        os.makedirs(uploads_dir, exist_ok=True)

        class_names_path = os.path.join(uploads_dir, 'class_names.csv')
        weights_path = None
        image_path = os.path.join(uploads_dir, 'image.jpg')

        if not class_names_file:
            raise ValueError("Class names file is required.")
        class_names_file.save(class_names_path)

        if not weights_file:
            raise ValueError("Weights file is required.")
         # Check if the model file is .keras or .h5 format
        original_filename = weights_file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        weights_path = os.path.join(uploads_dir, 'model_weights' + file_extension) # Use original file extension to save

        weights_file.save(weights_path)
        logging.info(f"Saved model with extension {file_extension} to {weights_path}")

        if not image_file:
            raise ValueError("Image file is required.")
        image_file.save(image_path)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # Catch other potential file errors
        return jsonify({"error": f"Error handling uploaded files: {e}"}), 500

    # Determine and execute the appropriate script
    try:
        if architecture in ['ResNet50', 'VGG16', 'InceptionV3']:
            script_name = 'gcamScript.py'  # Ensure this script exists in api folder
            args = [class_names_path, weights_path, architecture, image_path]
        elif architecture == 'Custom':
             # Check if the uploaded model is a .keras file
            if weights_file and weights_file.filename.endswith('.keras'):
                script_name = 'kerasScript.py'
            else:
                script_name = 'customScript.py'
            args = [class_names_path, weights_path, image_path]
        else:
            raise ValueError("Unsupported architecture type.")

        command = ['python', os.path.join(api_dir, script_name)] + args

        # Log the command being executed
        logging.info(f"Executing command: {' '.join(command)}")
        
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=api_dir  # Crucial: Set the working directory
        )
        
        # Log all outputs for debugging
        logging.info(f"Return code: {process.returncode}")
        logging.info(f"STDOUT: {process.stdout}")
        logging.info(f"STDERR: {process.stderr}")
        
        if process.returncode != 0:
            return jsonify({"error": f"Script execution failed with code {process.returncode}: {process.stderr}"}), 500
        
        # Attempt to parse JSON from stdout
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON output from Grad-CAM script. Parse error: {e}. Raw output: {process.stdout[:200]}..."}), 500

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        predicted_class_name = result["predicted_class_name"]
        visualization_image_path = result["visualization_path"]

        # Encode the image to base64
        try:
            with open(visualization_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            response = {
                "image": encoded_string,
                "predicted_class": predicted_class_name,
            }

            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": f"Error encoding or opening image: {e}"}), 500

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Error running Grad-CAM script: {e.stderr.strip()} (return code: {e.returncode})"}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


