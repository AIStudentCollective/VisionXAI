"""
LLM Explainer Module for VisionXAI

This module handles the generation of natural language explanations
for image classification results using vision-capable LLMs via OpenRouter.
"""

import os
import io
import base64
from PIL import Image
from typing import Optional, Dict, Any, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variables
api_key = os.environ.get("OPENROUTER_API_KEY")
logger.info(f"API Key present: {bool(api_key)}")
if not api_key or api_key == "<OPENROUTER_API_KEY>":
    logger.warning("OpenRouter API key not found in environment variables. Using fallback explanations.")
else:
    logger.info("OpenRouter API key found!")

# Try to import OpenAI but don't fail if it's not available
try:
    from openai import OpenAI
    # Initialize the OpenAI client with OpenRouter configuration if API key exists
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or "<PLACEHOLDER>"  # Use placeholder if no key is available
    )
    OPENAI_AVAILABLE = True
    logger.info("OpenAI module loaded successfully")
except ImportError as e:
    logger.warning(f"OpenAI module not available - using fallback explanations only. Error: {str(e)}")
    OPENAI_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error initializing OpenAI client: {str(e)}")
    OPENAI_AVAILABLE = False

def encode_image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return ""

def get_image_mime_type(image_path: str) -> str:
    """
    Determine the MIME type of an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: MIME type of the image
    """
    try:
        img = Image.open(image_path)
        format_to_mime = {
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'PNG': 'image/png',
            'GIF': 'image/gif',
            'WEBP': 'image/webp'
        }
        return format_to_mime.get(img.format, 'image/jpeg')
    except Exception as e:
        logger.error(f"Error determining image MIME type: {str(e)}")
        return "image/jpeg"  # Default to JPEG

def generate_explanation(
    original_image_path: str, 
    heatmap_image_path: str, 
    predicted_class: str, 
    architecture: str
) -> str:
    """
    Generate an explanation for the model's prediction using a vision-capable LLM via OpenRouter.
    
    Args:
        original_image_path (str): Path to the original input image
        heatmap_image_path (str): Path to the generated heatmap visualization
        predicted_class (str): The model's prediction
        architecture (str): The model architecture used
        
    Returns:
        str: An explanation of why the model made its prediction
    """
    try:
        # Check if we have a valid API key first
        if not api_key or api_key == "<OPENROUTER_API_KEY>":
            logger.info("No valid API key found, using fallback explanation")
            return get_fallback_explanation(predicted_class, architecture)
        
        # Check if OpenAI is available before trying to use it
        if not OPENAI_AVAILABLE:
            logger.info("OpenAI not available, using fallback explanation")
            return get_fallback_explanation(predicted_class, architecture)
        
        # Encode the images to base64
        original_image_base64 = encode_image_to_base64(original_image_path)
        heatmap_image_base64 = encode_image_to_base64(heatmap_image_path)
        
        if not original_image_base64 or not heatmap_image_base64:
            logger.error("Failed to encode images. Using fallback explanation.")
            return get_fallback_explanation(predicted_class, architecture)
        
        # Get image dimensions and basic visual features
        img = Image.open(original_image_path)
        width, height = img.size
        image_format = img.format
        image_mode = img.mode
        
        # Get MIME types
        original_mime = get_image_mime_type(original_image_path)
        heatmap_mime = get_image_mime_type(heatmap_image_path)
        
        # Log that we're generating an explanation
        logger.info(f"Generating explanation for class '{predicted_class}' using {architecture} model with vision LLM")
        
        # Prepare a detailed system prompt for the vision LLM
        system_prompt = f"""You are an expert in computer vision and machine learning, specializing in explaining AI model decisions.
        Your task is to analyze the provided images and explain how a {architecture} model classified the original image as '{predicted_class}'.
        
        You will be shown two images:
        1. The original image that was classified
        2. A heatmap visualization showing which parts of the image the model focused on when making its classification decision
        
        Explain the classification in a clear, educational way that helps non-technical users understand:
        1. What specific visual features the model detected that led to this classification
        2. How the heatmap reveals the model's attention and reasoning
        3. Why these features are characteristic signatures of the predicted class
        4. How confident this prediction seems and what factors influence that confidence
        
        Focus on the relationship between the highlighted areas in the heatmap and the visual features in the original image.
        Keep your language conversational but informative."""
        
        # Create a multimodal message structure with both images
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "text", 
                    "text": f"This is the original image that a {architecture} model classified as '{predicted_class}':"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{original_mime};base64,{original_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "And this is the attention heatmap showing which parts of the image the model focused on:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{heatmap_mime};base64,{heatmap_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Based on the original image and the attention heatmap, explain why this {architecture} model would classify this as '{predicted_class}'. What visual features is it detecting, and how does the attention map confirm this? Also, assess how confident this prediction seems based on the attention patterns."
                }
            ]}
        ]
        
        # Use a more widely available model (Claude 3 Sonnet instead of Opus)
        model = "anthropic/claude-3-sonnet:beta"  # More widely accessible than Opus
        
        logger.info(f"Attempting to call model: {model} with API key: {'[REDACTED]' if api_key else 'None'}")
        
        # Log a sample of the content being sent (for debugging)
        logger.info(f"Sending images with content types: {original_mime} and {heatmap_mime}")
        
        # Call OpenRouter with a vision-capable model
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://visionxai.ai",
                    "X-Title": "VisionXAI",
                },
                model=model,
                messages=messages,
                max_tokens=1024  # Adjust as needed
            )
            
            # Extract the explanation from the completion
            explanation = completion.choices[0].message.content
            logger.info("Successfully generated explanation from vision LLM")
            return explanation.strip()
            
        except Exception as api_error:
            logger.error(f"Error calling OpenRouter vision API: {str(api_error)}")
            # If the API call fails, return a detailed error message only during development
            if os.environ.get("FLASK_ENV") == "development":
                return f"API Error: {str(api_error)}\n\nFalling back to template response:\n\n{get_fallback_explanation(predicted_class, architecture)}"
            # In production, just return the fallback without error details
            return get_fallback_explanation(predicted_class, architecture)
        
    except Exception as e:
        # In case of an error, return a generic explanation
        logger.error(f"Error in generate_explanation: {str(e)}")
        return get_fallback_explanation(predicted_class, architecture)


def get_fallback_explanation(predicted_class: str, architecture: str) -> str:
    """
    Generate a fallback explanation when the LLM service is unavailable.
    
    Args:
        predicted_class (str): The predicted class name
        architecture (str): The model architecture used
        
    Returns:
        str: A generic explanation of the model's prediction
    """
    return f"""
    ## Explanation for {predicted_class} Classification
    
    The {architecture} model has classified this image as **{predicted_class}** based on visual features 
    detected in the image. 
    
    Key points in the explanation:
    
    1. The model identified specific visual patterns typically associated with {predicted_class}
    2. The attention map highlights the regions of the image that contributed most to this classification
    3. These highlighted areas contain distinctive features that the model has learned to associate with {predicted_class}
    4. The intensity of the highlights indicates the relative importance of each region to the final classification
    
    This explanation is a fallback response generated when the detailed LLM explanation service is unavailable.
    """