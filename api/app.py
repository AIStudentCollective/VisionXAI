from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_restful import Api, Resource
import os
import subprocess

app = Flask(__name__)
api = Api(app)

class DataHandler(Resource):
    def post(self):
        # Get uploaded files and architecture
        class_names_file = request.files.get('class_names')
        weights_file = request.files.get('weights')
        image_file = request.files.get('image')
        architecture = request.form.get('architecture')

        # Get the API directory (where app.py is located)
        api_dir = os.path.dirname(__file__)

        # File handling and validation
        try:
            class_names_path = os.path.join(api_dir, 'uploads/class_names.csv')
            weights_path = os.path.join(api_dir, 'uploads/model_weights.h5')
            image_path = os.path.join(api_dir, 'uploads/image.jpg')
            
            uploads_dir = os.path.join(api_dir, 'uploads')  # Uploads directory within the API directory
            os.makedirs(uploads_dir, exist_ok=True)

            if not class_names_file:
                raise ValueError("Class names file is required.")
            class_names_file.save(class_names_path)

            if not weights_file:
                raise ValueError("Weights file is required.")
            weights_file.save(weights_path)

            if not image_file:
                raise ValueError("Image file is required.")
            image_file.save(image_path)

        except ValueError as e:
            return Response(str(e), status=400, mimetype='text/plain')
        except Exception as e:  # Catch other potential file errors
            return Response(f"Error handling uploaded files: {e}", status=500, mimetype='text/plain')

        # Determine and execute the appropriate script
        try:
            if architecture in ['ResNet50', 'VGG16', 'InceptionV3']:
                script_name = 'gcamScript.py'  # Or script_path using os.path.join as before
                args = [class_names_path, weights_path, architecture, image_path]
            elif architecture == 'Custom':  # Example: Handle Custom architecture
                script_name = 'customScript.py' # Make sure this script exists
                args = [class_names_path, weights_path, image_path]  # Arguments for custom script
            else:
                raise ValueError("Unsupported architecture type.")

            command = ['python', script_name] + args

            process = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True, 
                cwd=api_dir  # Crucial: Set the working directory
            )

            output_lines = process.stdout.strip().split(',')
            if len(output_lines) != 2:
                raise ValueError("Unexpected output format from Grad-CAM script.")

            predicted_class_name = output_lines[0].strip()
            visualization_image_path = output_lines[1].strip()


        except subprocess.CalledProcessError as e:
            return Response(f"Error running Grad-CAM script: {e.stderr.strip()} (return code: {e.returncode})", status=500, mimetype='text/plain')
        except ValueError as e:
            return Response(str(e), status=400, mimetype='text/plain')
        except Exception as e:
            return Response(f"An unexpected error occurred: {str(e)}", status=500, mimetype='text/plain')

        # Serve the generated image  (send_file is more efficient)
        return send_file(visualization_image_path, mimetype='image/png')


api.add_resource(DataHandler, '/api/data')


if __name__ == '__main__':
    app.run(debug=True, port=5000)