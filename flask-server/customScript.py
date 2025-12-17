import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import json
import traceback

# Suppress TensorFlow progress bars and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=Debug, 1=Info, 2=Warning, 3=Error
tf.get_logger().setLevel('ERROR')  # Only show errors from TensorFlow

def get_target_size(model):
    try:
        input_shape = model.input_shape
        return (input_shape[1], input_shape[2])
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
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, tf.argmax(preds[0])]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]  # Get the output of the last convolutional layer
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]  # Calculate the heatmap
        heatmap = tf.squeeze(heatmap)  # Remove dimensions of size 1

        # Normalize heatmap to 0-1 range
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # Use tf.math.reduce_max
        return heatmap.numpy() # Convert to NumPy array
    except Exception as e:
        error_message = f"Error generating heatmap: {e}"
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
        error_message = f"Error overlaying heatmap: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)


def gradcam(model_path, img_path, csv_file):
    try:
        # Print model path to stderr for debugging (won't affect JSON output)
        print(f"Loading model from: {model_path}", file=sys.stderr)
        model = tf.keras.models.load_model(model_path, compile=False)  # Add compile=False to avoid warning
    except Exception as e:
        error_message = f"Error loading model: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

    img_array = load_and_preprocess_image(img_path, model)

    try:
        last_conv_layer_name = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        superimposed_img = overlay_heatmap(img_path, heatmap)
    except ValueError as e:
        error_message = f"Error generating heatmap: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_message = f"An unexpected error occurred during image processing: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

    visualization_filename = 'uploads/visualization.png'
    output_image_path = os.path.abspath(visualization_filename)

    try:
        cv2.imwrite(output_image_path, superimposed_img)
    except Exception as e:
        error_message = f"Error saving visualization: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

    try:
        # Suppress verbose output during prediction
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')  # Redirect stdout to /dev/null
        
        pred_index = np.argmax(model.predict(img_array, verbose=0))  # Set verbose=0 to silence prediction output
        
        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout
    except Exception as e:
        error_message = f"Error predicting with model: {e}"
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
        error_message = f"Error loading or processing class mapping file: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run Grad-CAM on an image using a custom architecture.')
    parser.add_argument('csv_file', type=str, help='CSV file with class mappings')
    parser.add_argument('model_path', type=str, help='Path to the custom model (.h5 file)')
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