import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

print("Starting application...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check if model file exists
model_files = ['deepfake_model.pth', 'deepfake_detector.pth', 'model.pth', 'best_model.pth']
available_files = []
for file in model_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        available_files.append(f"{file} ({size} bytes)")
        print(f"✅ Found model file: {file} (Size: {size} bytes)")

if not available_files:
    print("❌ No model files found!")
    print("Current directory contents:")
    for item in os.listdir('.'):
        print(f"  - {item}")

# Load the model
def load_model():
    print("Attempting to load model...")
    
    # Initialize the same model architecture as training
    try:
        model = models.resnet18(pretrained=False)  # Don't download pretrained weights
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # 2 classes: Real(0) and Fake(1)
        print(f"✅ Model architecture created. FC layer: {num_features} -> 2")
    except Exception as e:
        print(f"❌ Error creating model architecture: {e}")
        return None
    
    # Try to load the trained weights
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                print(f"Trying to load: {model_file}")
                
                # Load the state dictionary
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Handle different save formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and len(checkpoint) > 10:  # Likely state_dict
                    state_dict = checkpoint
                else:
                    print(f"Unexpected checkpoint format for {model_file}")
                    continue
                
                # Load state dict into model
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                
                print(f"✅ Model loaded successfully from {model_file}!")
                
                # Test the model with dummy input
                dummy_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    output = model(dummy_input)
                    print(f"✅ Model test successful. Output shape: {output.shape}")
                
                return model
                
            except Exception as e:
                print(f"❌ Error loading {model_file}: {str(e)}")
                continue
    
    print("❌ Failed to load any model file")
    return None

# Initialize model
print("Initializing model...")
model = load_model()

if model is None:
    print("❌ MODEL LOADING FAILED - App will show error messages")
else:
    print("✅ MODEL LOADED SUCCESSFULLY")

# Define preprocessing - MUST match training exactly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Match training normalization
])

def predict_image(image):
    if model is None:
        return "❌ Model not loaded! Please check if 'deepfake_detector.pth' is uploaded correctly.", 0.0
    
    if image is None:
        return "Please upload an image first!", 0.0
    
    try:
        print(f"Processing image: {image.size}, mode: {image.mode}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted image to RGB")
        
        # Apply preprocessing
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"Raw outputs: {outputs}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
        
        # Map prediction to class names
        # Assuming: 0 = Real, 1 = Fake (adjust if your training used different mapping)
        class_names = ['Real', 'Fake']
        prediction = class_names[predicted_class]
        confidence_percent = confidence * 100
        
        # Create result string
        if prediction == 'Fake':
            result = f"🚨 **DEEPFAKE DETECTED** 🚨\nConfidence: {confidence_percent:.2f}%"
        else:
            result = f"✅ **REAL IMAGE** ✅\nConfidence: {confidence_percent:.2f}%"
        
        print(f"Final result: {prediction} ({confidence_percent:.2f}%)")
        return result, confidence_percent
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, 0.0

def analyze_image(image):
    return predict_image(image)

# Create Gradio interface
print("Creating Gradio interface...")

with gr.Blocks(title="Deepfake Detection Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Deepfake Detection Tool
        
        Upload an image to detect whether it's **real** or **artificially generated (deepfake)**.
        
        **How it works:**
        - Upload any image (JPG, PNG, etc.)
        - The AI model analyzes the image for signs of manipulation
        - Get results with confidence scores
        
        **⚠️ Disclaimer:** This tool is for educational purposes. Results may not be 100% accurate.
        """
    )
    
    # Add model status indicator
    if model is None:
        gr.Markdown("### ❌ **Model Status: NOT LOADED**\nPlease check if the model file is uploaded correctly.")
    else:
        gr.Markdown("### ✅ **Model Status: LOADED AND READY**")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Image for Analysis",
                height=400
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary",
                size="lg"
            )
            
        with gr.Column():
            result_output = gr.Markdown(
                value="Upload an image and click 'Analyze Image' to get started!"
            )
            
            confidence_output = gr.Number(
                label="Confidence Score (%)",
                precision=2,
                interactive=False
            )
    
    # Connect the button to the function
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[result_output, confidence_output]
    )
    
    # Also trigger on image upload
    image_input.change(
        fn=analyze_image,
        inputs=image_input,
        outputs=[result_output, confidence_output]
    )

print("✅ Gradio interface created")

# Launch the app
if __name__ == "__main__":
    print("Launching application...")
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )