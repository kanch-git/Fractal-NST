import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from PIL import Image
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
IMAGE_SIZE = 256
EPOCHS = 100
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e2

content_dir = "content"
style_dir = "style"
output_dir = "nst_output"
os.makedirs(output_dir, exist_ok=True)

# Image Loader
loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

content_img = load_image(os.path.join(content_dir, random.choice(os.listdir(content_dir))))
style_img   = load_image(os.path.join(style_dir, random.choice(os.listdir(style_dir))))

# VGG19 Model
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# Layers
content_layer = "21"  # conv4_2
style_layers = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "28": "conv5_1"
}

def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name == content_layer:
            features["content"] = x
        if name in style_layers:
            features[style_layers[name]] = x
    return features

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)

# Extract targets WITHOUT gradients
with torch.no_grad():
    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)
    style_grams = {layer: gram_matrix(style_features[layer])
                   for layer in style_layers.values()}

# Target Image
target = content_img.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target], lr=0.02)

# NST Optimization
for epoch in range(1, EPOCHS + 1):

    target_features = get_features(target, vgg)

    # Content loss (conv block)
    content_loss = torch.mean(
        (target_features["content"] - content_features["content"]) ** 2
    )

    # Style loss
    style_loss = 0
    for layer in style_layers.values():
        target_gram = gram_matrix(target_features[layer])
        style_loss += torch.mean((target_gram - style_grams[layer]) ** 2)

    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{EPOCHS}] | Total Loss: {total_loss.item():.4f}")

# Save Output
output_path = os.path.join(output_dir, "nst_result.png")
transforms.ToPILImage()(target.squeeze().clamp(0, 1).cpu()).save(output_path)

print("NST completed successfully.")
