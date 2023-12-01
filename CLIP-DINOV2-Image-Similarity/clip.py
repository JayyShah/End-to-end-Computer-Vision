import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

#Extract features from image1
image1 = Image.open('img1.jpg')
with torch.no_grad():
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    image_features1 = model.get_image_features(**inputs1)

#Extract features from image2
image2 = Image.open('img2.jpg')
with torch.no_grad():
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    image_features2 = model.get_image_features(**inputs2)

#Compute their cosine similarity and convert it into a score between 0 and 1
cos = nn.CosineSimilarity(dim=0)
sim = cos(image_features1[0],image_features2[0]).item()
sim = (sim+1)/2
print('Similarity:', sim)