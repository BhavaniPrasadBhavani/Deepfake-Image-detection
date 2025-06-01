# 🔍 Deepfake Image detector

An AI-powered system that identifies whether images are authentic or AI-generated (deepfakes) using deep learning techniques.

## 🌟 Features

- **High Accuracy Detection**: 98%+ accuracy in identifying deepfakes
- **Real-time Analysis**: Instant image upload and classification
- **User-friendly Interface**: Simple web-based Gradio interface
- **Live Demo**: Publicly accessible URL for testing
- **Cost-effective Deployment**: Free GPU access via Hugging Face Spaces

## 🚀 Live Demo

https://huggingface.co/spaces/SkyNoel/deepfake_image

Upload any image and get instant results on whether it's authentic or AI-generated.

## 🛠️ Tech Stack

### Current Implementation (Hugging Face)
- **Platform**: Hugging Face Spaces
- **Interface**: Gradio
- **Models**: Google Colab manully trained model
- **Language**: Python
- **Deployment**: Automated via HF Spaces

## 📊 Performance Metrics

- **Accuracy**: 98% (achieved in 5 epochs with ResNet)
- **Deployment Time**: < 5 minutes on Hugging Face Spaces
- **Cost**: $0 (using free HF GPU hours)

## 🏗️ Project Architecture



## 🚀 Quick Start

### Option 1: Use Live Demo
1. Visit the "https://huggingface.co/spaces/SkyNoel/deepfake_image"
2. Upload an image
3. Get instant results!

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detection
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Option 3: Deploy on Hugging Face Spaces
1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your repository
4. Automatic deployment will handle the rest!

## 📁 Project Structure


## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection
   cd deepfake-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 📋 Requirements

```txt
gradio>=3.40.0
torch>=1.12.0
torchvision>=0.13.0
transformers>=4.21.0
Pillow>=9.0.0
numpy>=1.21.0
requests>=2.28.0
```

## 🎯 Use Cases

- **Media Verification**: Authenticate news and social media content
- **Security Systems**: Detect manipulated identification documents
- **Digital Forensics**: Support investigation of fraudulent visual evidence
- **Content Moderation**: Automated detection of synthetic media
- **Educational Tool**: Demonstrate deepfake detection techniques

## 🧪 Testing

Upload test images to verify the system works correctly:

1. **Real Images**: Should be classified as "Authentic"
2. **AI-Generated Images**: Should be classified as "Deepfake"
3. **Edge Cases**: Low quality, filtered, or edited images

## 🔄 Project Evolution

This project evolved from an AWS-based Flask API to a Hugging Face Spaces deployment:

| Aspect | AWS Approach | Hugging Face Approach |
|--------|-------------|---------------------|
| **Cost** | ~$50-100/month | Free |
| **Deployment Time** | Hours | Minutes |
| **Maintenance** | High | Minimal |
| **Scalability** | Manual | Automatic |
| **User Interface** | Custom Flask | Gradio |

## 🚧 Future Enhancements

- [ ] **Video Deepfake Detection**: Extend to analyze video content
- [ ] **Batch Processing**: Handle multiple images simultaneously
- [ ] **API Endpoints**: Provide REST API for integration
- [ ] **Mobile App**: React Native or Flutter implementation
- [ ] **Advanced Models**: Experiment with Vision Transformers
- [ ] **Explainable AI**: Show which features indicate deepfakes

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution
- Model improvements and optimization
- User interface enhancements
- Additional preprocessing techniques
- Documentation improvements
- Bug fixes and testing


##  Acknowledgments

- **Hugging Face** for providing free GPU access and deployment platform
- **PyTorch Team** for the deep learning framework
- **Gradio Team** for the simple UI framework
- **ResNet Authors** for the foundational architecture
- **Open Source Community** for pre-trained models and datasets

## 📞 Contact

**Kumbaganti Bhavani Prasad** - [kumbagantibhavaniprasad@example.com]

**LinkedIn**: [https://www.linkedin.com/in/kbhavaniprasad/]

**Project Link**: [https://github.com/BhavaniPrasadBhavani/Deepfake-Image-detection]

**Live Demo**: [https://huggingface.co/spaces/SkyNoel/deepfake_image]

---

⭐ **Star this repository if you found it helpful!**

#MachineLearning #DeepLearning #ComputerVision #AI #HuggingFace #Gradio #Python
