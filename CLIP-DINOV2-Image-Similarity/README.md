- In the world of artificial intelligence, two giants stand tall in the realm of computer vision: CLIP and DINOv2.**CLIP** revolutionized **image understanding**, while **DINOv2** brought a fresh approach to **self-supervised learning**.

- In this Repository We aim to discover which of these models truly excels in the world of image similarity tasks.

- Ensure you have the necessary packages installed. It is advisable to set up and utilize a virtual environment.

# Image Similarity with CLIP
- Calculating the similarity between two images with CLIP is a straightforward process, achieved in just two steps: first, extract the features of both images, and then compute their cosine similarity.

*Check Clip.py for Implementation*

# Image Similarity with DINOv2

- The process of computing similarity between two images with DINOv2 mirrors that of CLIP.
- Utilizing DINOv2 requires the same set of packages as previously mentioned, without the need for any additional installations.

*Check dinov2.py for Implementation*

# Testing with the COCO dataset

- The process we employ is as follows:
1. Iterate through the dataset to extract the features of all the images.
2. Store the embeddings in a FAISS index.
3. Extract the features of an input image.
4. Retrieve the top-three similar images.

*pip install faiss-[gpu|cpu]* - As we need to store Embeddings.

- In this small subset, it appears that DINOv2 demonstrates a slightly superior performance.

# Benchmarking against the DISC21 Dataset

# Dataset

To benchmark CLIP and DINOv2, we have chosen the DISC21 dataset, purposefully created for image similarity searches. Due to its substantial size of 350GB, we will be using a subset of 150.000 images.

- Reuse the scripts above to extract features and then compute image similarity.

# Metrics employed
In terms of metrics, we will calculate:

**Accuracy**: the ratio of correctly predicted images to the total number of images.

**Top-3 Accuracy**: the ratio of times the correct image is found within the top three similar images to the total number of images.

**Computational time**: the time required to process the entire dataset.

# Outcome of the benchmark

## Features extraction

**CLIP**: 70.7 images per second

**DINOv2**: 69.7 images per second

# Analysis
DINOv2 emerges as the clear frontrunner, achieving an impressive accuracy of 64% on a notably challenging dataset. By contrast, CLIP demonstrates a more modest accuracy, reaching 28.45%.

Regarding computational efficiency, both models exhibit remarkably similar feature extraction times. This parity places neither model at a distinct advantage in this regard.

# Limitations
While this benchmark offers valuable insights, it’s crucial to recognize its limitations. The evaluation was conducted on a subset of 1448 images, compared against a pool of 150,000 images. Given the entire dataset’s size of 2.1 million images, this narrowed focus was necessary to conserve resources.

It’s worth noting that MetaAI employs the DISC21 dataset as a benchmark for its model, potentially giving DINOv2 a favorable advantage. However, our tests on the COCO dataset revealed intriguing nuances: DINOv2 displays a heightened ability to identify primary elements in an image, whereas CLIP demonstrates adeptness at focusing on specific details within an input image (as exemplified by the image of the bus).

Lastly, it’s essential to consider the difference in embedding dimensions between CLIP and DINOv2. CLIP utilizes an embedding dimension of 512, whereas DINOv2 operates with 768. While an alternative could be to employ the larger CLIP model with a matching embedding dimension, it’s worth noting that this comes at the cost of speed. A quick test on a small subset showed a slight performance boost but without achieving the level of performance demonstrated by DINOv2.

# Conclusion

DINOv2 demonstrates superior accuracy in image similarity tasks, showcasing its potential for practical applications. CLIP, while commendable, falls short in comparison. It’s worth noting that CLIP can be particularly useful in scenarios that demand a focus on small details. Both models exhibit similar computational efficiency, making the choice task-specific.