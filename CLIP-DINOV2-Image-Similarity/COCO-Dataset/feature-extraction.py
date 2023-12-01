import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load CLIP model and processor
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

#Load DINOv2 model and processor
processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

#Retrieve all filenames
images = []
for root, dirs, files in os.walk('./val2017/'):
    for file in files:
        if file.endswith('jpg'):
            images.append(root  + '/'+ file)


#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor_clip(images=image, return_tensors="pt").to(device)
        image_features = model_clip.get_image_features(**inputs)
        return image_features

def extract_features_dino(image):
    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to(device)
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)

#Create 2 indexes.
index_clip = faiss.IndexFlatL2(512)
index_dino = faiss.IndexFlatL2(768)

#Iterate over the dataset to extract features X2 and store features in indexes
for image_path in images:
    img = Image.open(image_path).convert('RGB')
    clip_features = extract_features_clip(img)
    add_vector_to_index(clip_features,index_clip)
    dino_features = extract_features_dino(img)
    add_vector_to_index(dino_features,index_dino)

#store the indexes locally
faiss.write_index(index_clip,"clip.index")
faiss.write_index(index_dino,"dino.index")