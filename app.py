import gradio as gr
import torch
import numpy as np
from PIL import Image
import albumentations
import pandas as pd
from lightning_model import LitClassification

# Load class labels
df = pd.read_csv("imagenet_class_labels.csv")
class_labels = df['Labels'].tolist()

# Initialize model and load checkpoint
model = LitClassification()
checkpoint = torch.load("bestmodel-epoch=46-monitor-val_acc1=63.7760009765625.ckpt", 
                       map_location=torch.device('cpu'))  # Load to CPU by default
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Image preprocessing
valid_aug = albumentations.Compose(
    [
        albumentations.Resize(224, 224, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array
    image = np.array(image)
    
    # Center crop 95% area
    H, W, C = image.shape
    image = image[int(0.04 * H) : int(0.96 * H), int(0.04 * W) : int(0.96 * W), :]
    
    # Apply augmentations
    augmented = valid_aug(image=image)
    image = augmented["image"]
    
    # Convert to tensor and add batch dimension
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0)
    return image

def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Convert predictions to labels and probabilities
    results = {
        class_labels[idx]: float(prob)
        for prob, idx in zip(top5_prob[0], top5_indices[0])
    }
    
    return results

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=["sample_imgs/stock-photo-large-hot-dog.jpg"],
    title="ImageNet Classification with ResNet50",
    description="Upload an image to classify it into one of 1000 ImageNet categories."
)

if __name__ == "__main__":
    iface.launch()
