# Import Dependencies

import base64
import os
from io import BytesIO
import cv2
import faiss
import numpy as np
import requests
from PIL import Image
import json
import supervision as sv

# Download the Dataset from RoboFlow

import roboflow

roboflow.login()

roboflow.download_dataset(
    dataset_url="https://universe.roboflow.com/team-roboflow/coco-128/dataset/2",
    model_format="coco")

"""
When you run this code, you will first be asked to authenticate if you have not already signed
in to Roboflow via the command line. You only need to run this code once to download your dataset, 
so it does not need to be part of your main script.
"""

# Calculate CLIP vectors for Images

INFERENCE_ENDPOINT = "http://localhost:9001"

def get_image_embedding(image: str) -> dict:
    image = image.convert("RGB")
    
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    payload = {
        "body": API_KEY,
        "image": {"type": "base64", "value": image},
    }
    
    data = requests.post(
    	INFERENCE_ENDPOINT + "/clip/embed_image?api_key=" + API_KEY, json=payload
    )
    
    response = data.json()
    embedding = response["embeddings"]
    return embedding

# Create a Vector Database

index = faiss.IndexFlatL2(512)
file_names = []
TRAIN_IMAGES = os.path.join(DATASET_PATH, "train")

for frame_name in os.listdir(TRAIN_IMAGES):
    try:
        frame = Image.open(os.path.join(TRAIN_IMAGES, frame_name))
    except IOError:
        print("error computing embedding for", frame_name)
        continue

    embedding = get_image_embedding(frame)
    
    index.add(np.array(embedding).astype(np.float32))
    
    file_names.append(frame_name)

 faiss.write_index(index, "index.bin")
 
 with open("index.json", "w") as f:
 	json.dump(file_names, f)

"""
In this code, we create an index that is stored in a local file. This index stores all of our embeddings. 
We also make a list of the order in which files were inserted, which is needed because to map our vectors
back to the images they represent.

We then save the index to a file called “index.bin”. 
We also store a mapping between the position in which images were inserted into the index and the names of files 
that the position represents. This is needed to map the insertion position, which our index uses, back to a 
filename if we want to re-use our index next time we run the program.

"""

# Search the Database

FILE_NAME = ""
DATASET_PATH = ""
RESULTS_NUM = 3

query = get_image_embedding(Image.open(FILE_NAME))
D, I = index.search(np.array(query).astype(np.float32), RESULTS_NUM)

images = [cv2.imread(os.path.join(TRAIN_IMAGES, file_names[i])) for i in I[0]]

sv.plot_images_grid(images, (3, 3))

"""
In the code above, replace FILE_NAME with the name of the image that you want to use in your search.
Replace DATASET_PATH with the path where the images for which you calculated embeddings earlier are stored
(i.e. COCO-128-2/train/). This code returns three results by default, but you can return more or less by 
replacing the value of RESULTS_NUM.

This code will calculate an embedding for a provided image, which is then used as a search query with our vector 
database. We then plot the top three most similar images.
"""

