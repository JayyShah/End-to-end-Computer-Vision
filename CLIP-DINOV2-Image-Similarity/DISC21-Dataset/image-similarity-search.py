import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
from PIL import Image
import os

#Input image
source='laptop.jpg'
image = Image.open(source)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load model and processor DINOv2 and CLIP
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

#Extract features for CLIP
with torch.no_grad():
    inputs_clip = processor_clip(images=image, return_tensors="pt").to(device)
    image_features_clip = model_clip.get_image_features(**inputs_clip)

#Extract features for DINOv2
with torch.no_grad():
    inputs_dino = processor_dino(images=image, return_tensors="pt").to(device)
    outputs_dino = model_dino(**inputs_dino)
    image_features_dino = outputs_dino.last_hidden_state
    image_features_dino = image_features_dino.mean(dim=1)

def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

image_features_dino = normalizeL2(image_features_dino)
image_features_clip = normalizeL2(image_features_clip)

#Search the top 5 images
index_clip = faiss.read_index("clip.index")
index_dino = faiss.read_index("dino.index")

#Get distance and indexes of images associated
d_dino,i_dino = index_dino.search(image_features_dino,5)
d_clip,i_clip = index_clip.search(image_features_clip,5)