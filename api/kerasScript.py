import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import json
import traceback
import contextlib
from io import StringIO

# Suppress TensorFlow progress bars and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=Debug, 1=Info, 2=Warning, 3=Error
tf.get_logger().setLevel('ERROR')  # Only show errors from TensorFlow

@contextlib.contextmanager
def suppress_stdout():
    """
    A context manager that redirects stdout to /dev/null
    """
    save_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout

def get_target_size(model):
    try:
        input_shape = model.input_shape
        # Check if input_shape is a tuple or a InputLayer object
        if isinstance(input_shape, tuple):
            return (input_shape[1], input_shape[2])
        elif hasattr(model, 'layers') and model.layers and hasattr(model.layers[0], 'input_shape'):
            # Handle case where model is a Sequential model
            shape = model.layers[0].input_shape
            if shape and len(shape) > 2:
                return (shape[1], shape[2])
            # Try to look at the first layer's batch_input_shape
            batch_shape = getattr(model.layers[0], 'batch_input_shape', None)
            if batch_shape and len(batch_shape) > 2:
                return (batch_shape[1], batch_shape[2])
        
        # Default fallback to 128x128 (based on your model)
        print("Warning: Could not determine input shape, using default 128x128", file=sys.stderr)
        return (128, 128)
    except AttributeError as e:
        error_message = f"Model input shape missing. Please check the format of your Keras model. {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

def load_and_preprocess_image(img_path, model):
    try:
        target_size = get_target_size(model)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0
    except FileNotFoundError as e:
        error_message = f"Error: Image file not found: {img_path}. {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_message = f"Error loading or preprocessing image: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    error_message = "Could not find a convolutional layer."
    print(json.dumps({"error": error_message}), file=sys.stderr)
    sys.exit(1)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    try:
        # First, predict with the model to ensure it's built
        with suppress_stdout():
            predictions = model.predict(img_array, verbose=0)
        pred_index = np.argmax(predictions[0])
        
        # Create a model that goes from the input to the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_index = [i for i, layer in enumerate(model.layers) if layer.name == last_conv_layer_name][0]
        
        # Create partial models
        conv_model = tf.keras.models.Sequential(model.layers[:last_conv_index+1])
        
        # Get the output of the last conv layer
        with suppress_stdout():
            last_conv_output = conv_model.predict(img_array)
        
        # Create a Gradient model
        with tf.GradientTape() as tape:
            # Cast the conv output tensor to a tf.Variable
            conv_output = tf.Variable(last_conv_output)
            
            # Create a model from last conv to predictions
            remaining_layers = model.layers[last_conv_index+1:]
            pred = conv_output
            for layer in remaining_layers:
                pred = layer(pred)
            
            # Get class output
            class_output = pred[:, pred_index]
        
        # Get gradients
        grads = tape.gradient(class_output, conv_output)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Convert to numpy for easier manipulation
        pooled_grads_np = pooled_grads.numpy()
        
        # Fix broadcasting by reshaping pooled_grads to match the last dimension of last_conv_output
        # Print shapes for debugging
        print(f"Last conv output shape: {last_conv_output.shape}", file=sys.stderr)
        print(f"Pooled grads shape before reshape: {pooled_grads_np.shape}", file=sys.stderr)
        
        # Get the first element of the batch (we only have one image)
        last_conv_output_np = last_conv_output[0]  # Shape should be (height, width, channels)
        
        # Reshape pooled_grads to match broadcasting requirements
        # We need to reshape it to (1, 1, channels) for proper broadcasting
        pooled_grads_reshaped = pooled_grads_np.reshape((1, 1, -1))
        
        print(f"Last conv output shape after getting first element: {last_conv_output_np.shape}", file=sys.stderr)
        print(f"Pooled grads shape after reshape: {pooled_grads_reshaped.shape}", file=sys.stderr)
        
        # Multiply each channel in the feature map by importance
        heatmap = np.sum(last_conv_output_np * pooled_grads_reshaped, axis=-1)
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8) # Adding small epsilon for numerical stability
        
        return heatmap
        
    except Exception as e:
        error_message = f"Error generating heatmap: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)


def overlay_heatmap(img_path, heatmap):
    try:
        img = cv2.imread(img_path)
        if img is None:
            error_message = f"Could not read image file: {img_path}"
            print(json.dumps({"error": error_message}), file=sys.stderr)
            sys.exit(1)
        heatmap = np.uint8(255 * heatmap)  # Convert to uint8 BEFORE resizing
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img.astype(np.uint8), 0.6, heatmap_colored.astype(np.uint8), 0.4, 0)
        return superimposed_img
    except FileNotFoundError as e:
        error_message = f"Error: Image file not found: {img_path}. {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_message = f"Error overlaying heatmap: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

def gradcam(model_path, img_path, csv_file):
    try:
        # Detect model format based on file extension
        file_extension = os.path.splitext(model_path)[1].lower()
        print(f"Loading model from: {model_path} with extension {file_extension}", file=sys.stderr)

        # Load the model based on its format
        if file_extension == '.keras':
            model = tf.keras.models.load_model(model_path, compile=False)
        elif file_extension == '.h5':
            model = tf.keras.models.load_model(model_path, compile=False)
        else:
             raise ValueError(f"Unsupported model format: {file_extension}. Only .keras and .h5 are allowed.")

        # Process the image
        img_array = load_and_preprocess_image(img_path, model)
        
        # Get predictions first to ensure the model is built
        with suppress_stdout():
            predictions = model.predict(img_array, verbose=0)
        pred_index = np.argmax(predictions[0])

    except Exception as e:
        error_message = f"Error loading model: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

    try:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"Using convolutional layer: {last_conv_layer_name}", file=sys.stderr)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        superimposed_img = overlay_heatmap(img_path, heatmap)
    except ValueError as e:
        error_message = f"Error generating heatmap: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_message = f"An unexpected error occurred during image processing: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

    visualization_filename = 'uploads/visualization.png'
    output_image_path = os.path.abspath(visualization_filename)

    try:
        cv2.imwrite(output_image_path, superimposed_img)
    except Exception as e:
        error_message = f"Error saving visualization: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

    class_mapping = load_class_mapping(csv_file)
    predicted_class_name = class_mapping.get(pred_index, "Unknown")

    return predicted_class_name, output_image_path

def load_class_mapping(csv_file):
    try:
        df = pd.read_csv(csv_file)
        class_mapping = {row['index']: row['class'] for _, row in df.iterrows()}
        return class_mapping
    except FileNotFoundError as e:
        error_message = f"Error: Class mapping file not found: {csv_file}. {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        error_message = f"Error: Class mapping file is empty: {csv_file}. {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as e:
        error_message = f"Error: Could not parse class mapping file: {csv_file}. {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        error_message = f"Error: Missing column ('index' or 'class') in class mapping file: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_message = f"Error loading or processing class mapping file: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run Grad-CAM on an image using a custom architecture.')
    parser.add_argument('csv_file', type=str ,help='CSV file with class mappings')
    parser.add_argument('model_path', type=str ,help='Path to the custom model (.h5 or .keras file)')
    parser.add_argument('image_path', type=str, help='Path to the image for inference')

    args = parser.parse_args()

    result = {} # Initialize result
    try:
        predicted_class_name, visualization_path = gradcam(args.model_path, args.image_path, args.csv_file)

        result = {
            "predicted_class_name": predicted_class_name,
            "visualization_path": visualization_path
        }

    except Exception as e:
        error_message = f"An error occurred in main: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

    # Make sure to redirect TF warnings/info to stderr instead of stdout
    tf.get_logger().setLevel('ERROR')

    # Clear any previous output to stdout to avoid mixing text
    sys.stdout.flush()

    # Send a single, complete JSON output
    print(json.dumps(result))

if __name__== '__main__':
    main()
