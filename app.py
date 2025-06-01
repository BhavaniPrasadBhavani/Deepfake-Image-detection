import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from model import DeepfakeResNet
import os

# Initialize model
model = DeepfakeResNet()
model.eval()

# Disable gradients for faster inference
torch.set_grad_enabled(False)

# Load model weights
try:
    model_path = "deepfake_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Preprocessing - matching training configuration
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Matching training normalization
                         std=[0.5, 0.5, 0.5])
])

# Inference function
def predict(img):
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = transform(img).unsqueeze(0)
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()
        label = "Fake" if pred == 1 else "Real"
        return f"{label} (Confidence: {confidence:.2%})"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(label="Prediction"),
    title="Deepfake Image Detector",
    description="Upload a face image to detect if it's real or AI-generated (deepfake).",
    examples=[["example_real.jpg"], ["example_fake.jpg"]] if os.path.exists("example_real.jpg") else None
)

if __name__ == "__main__":
    interface.launch() 