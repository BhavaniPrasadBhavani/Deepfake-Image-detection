import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Load the model
def load_model():
    # Initialize the same model architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Real and Fake
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load('deepfake_detector.pth', map_location='cpu'))
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

# Define the same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(image):
    if model is None:
        return "Model not loaded properly!", 0.0
    
    try:
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map prediction to class names (assuming 0=Fake, 1=Real)
        class_names = ['Fake', 'Real']
        prediction = class_names[predicted_class]
        
        # Create result string with confidence
        result = f"Prediction: {prediction}"
        confidence_percent = confidence * 100
        
        # Add color coding for the interface
        if prediction == 'Fake':
            result = f"🚨 **DEEPFAKE DETECTED** 🚨\nConfidence: {confidence_percent:.2f}%"
        else:
            result = f"✅ **REAL IMAGE** ✅\nConfidence: {confidence_percent:.2f}%"
        
        return result, confidence_percent
        
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0

def analyze_image(image):
    if image is None:
        return "Please upload an image first!", 0.0
    
    result, confidence = predict_image(image)
    return result, confidence

# Create Gradio interface
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
                label="Analysis Result",
                value="Upload an image and click 'Analyze Image' to get started!"
            )
            
            confidence_output = gr.Number(
                label="Confidence Score (%)",
                precision=2,
                interactive=False
            )
    
    # Example images section
    gr.Markdown("### 📸 Try with example images:")
    gr.Examples(
        examples=[
            ["example1.jpg"],
            ["example2.jpg"],
        ],
        inputs=image_input,
        label="Sample Images"
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

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )