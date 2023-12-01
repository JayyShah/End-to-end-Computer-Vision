from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)


image1 = Image.open('img1.jpg')
with torch.no_grad():
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    outputs1 = model(**inputs1)
    image_features1 = outputs1.last_hidden_state
    image_features1 = image_features1.mean(dim=1)

image2 = Image.open('img2.jpg')
with torch.no_grad():
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    outputs2 = model(**inputs2)
    image_features2 = outputs2.last_hidden_state
    image_features2 = image_features2.mean(dim=1)

cos = nn.CosineSimilarity(dim=0)
sim = cos(image_features1[0],image_features2[0]).item()
sim = (sim+1)/2
print('Similarity:', sim)