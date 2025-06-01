import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from model import DeepfakeResNet

# Load model
model = DeepfakeResNet()
model.load_state_dict(torch.load("deepfake_model.pth", map_location=torch.device("cpu")))
model.eval()

# Disable gradients for faster inference
torch.set_grad_enabled(False)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

# Inference function
def predict(img):
    img = transform(img).unsqueeze(0)
    output = model(img)
    pred = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred].item()
    label = "Fake" if pred == 1 else "Real"
    return f"{label} (Confidence: {confidence:.2%})"

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(label="Prediction"),
    title="Deepfake Image Detector",
    description="Upload a face image to detect if it's real or AI-generated (deepfake)."
)

if __name__ == "__main__":
    interface.launch() 