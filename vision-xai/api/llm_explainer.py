"""
LLM Explainer Module for VisionXAI

This module handles the generation of natural language explanations
for image classification results using OpenRouter and Grok-3-mini LLM.
"""

import os
import io
import base64
from PIL import Image
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variables
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key or api_key == "<OPENROUTER_API_KEY>":
    logger.warning("OpenRouter API key not found in environment variables. Using fallback explanations.")

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
except ImportError:
    logger.warning("OpenAI module not available - using fallback explanations only")
    OPENAI_AVAILABLE = False

def generate_explanation(
    original_image_path: str, 
    heatmap_image_path: str, 
    predicted_class: str, 
    architecture: str
) -> str:
    """
    Generate an explanation for the model's prediction using OpenRouter's LLM API.
    
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
            
        # Get image dimensions and basic visual features
        img = Image.open(original_image_path)
        width, height = img.size
        image_format = img.format
        image_mode = img.mode
        
        # Get heatmap image info
        heatmap_img = Image.open(heatmap_image_path)
        
        # Log that we're generating an explanation
        logger.info(f"Generating explanation for class '{predicted_class}' using {architecture} model")
        
        # Prepare a detailed system prompt for the LLM
        system_prompt = f"""You are an expert in computer vision and machine learning, specializing in explaining AI model decisions.
        Your task is to analyze how a {architecture} model classified an image as '{predicted_class}'.
        
        The user will provide information about an image and a heatmap visualization showing which parts of the image
        the model focused on when making its classification decision.
        
        Explain the classification in a clear, educational way that helps non-technical users understand:
        1. What specific visual features the model detected that led to this classification
        2. How the heatmap reveals the model's attention and reasoning
        3. Why these features are characteristic signatures of the predicted class
        4. How confident this prediction seems and what factors influence that confidence
        
        Keep your language conversational but informative."""
        
        # Prepare the user prompt with image properties
        user_prompt = f"""
        The {architecture} model classified an image as '{predicted_class}'.
        
        Image properties:
        - Dimensions: {width}x{height} pixels
        - Format: {image_format}
        - Color mode: {image_mode}
        
        The model's attention heatmap shows it's focusing most intensely on certain regions of the image.
        
        Please explain why this model would classify this as a '{predicted_class}', what visual features it's detecting,
        and how the attention map confirms this analysis. Also, assess how confident this prediction seems based on the
        available information.
        """
        
        # Check if OpenAI is available before trying to use it
        if not OPENAI_AVAILABLE:
            logger.info("OpenAI not available, using fallback explanation")
            return get_fallback_explanation(predicted_class, architecture)
            
        # Call OpenRouter with Grok-3-mini model if OpenAI is available
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://visionxai.ai",
                    "X-Title": "VisionXAI",
                },
                model="x-ai/grok-3-mini-beta",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract the explanation from the completion
            explanation = completion.choices[0].message.content
            logger.info("Successfully generated explanation from LLM")
            return explanation.strip()
            
        except Exception as api_error:
            logger.error(f"Error calling OpenRouter API: {str(api_error)}")
            # If the API call fails, return a detailed error message only during development
            if os.environ.get("FLASK_ENV") == "development":
                return f"API Error: {str(api_error)}\n\nFalling back to template response:\n\n{get_fallback_explanation(predicted_class, architecture)}"
            # In production, just return the fallback without error details
            return get_fallback_explanation(predicted_class, architecture)
        
    except Exception as e:
        # In case of an error, return a generic explanation
        print(f"Error calling LLM API: {str(e)}")
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
    