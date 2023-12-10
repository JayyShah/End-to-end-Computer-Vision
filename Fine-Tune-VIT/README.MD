# Fine-tuning image classification models from image search

## Easily fine-tune a ViT with images from Bing search and visualize the results in 5 lines of code

- The use of pre-trained models on large datasets, such as ImageNet, followed by fine-tuning on specific target datasets, has become the default approach in image classification. This has radically simplified the task of image classification.

- TODO: Assemble an image dataset using a customizable search that can integrate our own data with Bing, then fine-tune a model, and evaluate and debug it using Spotlight for interactive visualization — all in just five lines of code.

## Dependencies

`!pip install renumics-spotlight sliceguard[all]`

*Check the main.py for code example*

### Code BreakDown

1. Acquiring Images

- You can create your own dataset using BING image search. Adapting it is straightforward; just modify the class_names in the list. Internally, Sliceguard employs bing-image-downloader to asynchronously search and download images, but it's enhanced to filter images by license. For this example, we're using images that are "Free to share and use".

- Afterward, the df comprises data for 25 images from each class. Each row provides the path to the image on your storage, the label in text, the label as a numeral, and specifies whether the entry belongs to the training or test split:

2. Fine-Tuning with the Aid of Pretrained Transformers

- Once the dataset is ready, we employ sliceguard for fine-tuning, encapsulating the procedure as detailed in the Hugging Face Documentation using On a GeForce RTX 4070 Ti with 12 GB, this process takes less than 2 minutes. While we use the google/vit-base-patch16–224-in21k Vision Transformer (ViT) model [1] here, you have the option to select different models from the Hugging Face hub.

3. Enrich Your Data

- We enrich our dataset with embeddings and classification results with:

    - Prediction: The model’s output indicating the most likely class for each image.
    - Probabilities: The likelihood scores assigned by the model for each possible class, offering a measure of the model’s certainty.
    - Embeddings: These are dense vector representations of the images.

- We’ll utilize the enriched data in the next step.

4 Visualize data and model results

- Finally, we use Spotlight, a tool that assists in understanding and visualizing your data effectively, to examine the results and detect problematic clusters.

- We can interactively explore the dataset using this tool. By selecting image clusters in the similarity map, we can closely inspect them. Doing so reveals four key insights:

- The Royal Gamma Fish class has an outlier cluster featuring images of royalty and a Minecraft screenshot.

- Additionally, the Royal Gamma Fish class reveals two clusters featuring fish distinct from its namesake, along with a sizable school of fish.

- The Blue Tang class incorporates artistic drawings that feature a Blue Tang.

- To enhance the dataset’s quality, it’s advisable to remove these outliers and substitute them with better images.

5 Conclusion

- Image classification has been made easier with the use of pre-trained models


