from renumics import spotlight
from sliceguard.data import create_imagedataset_from_bing
from sliceguard.models.huggingface import finetune_image_classifier, generate_image_pred_probs_embeddings
from sliceguard.embeddings import generate_image_embeddings

# Create an Image Dataset from Bing Image Search
class_names = ["Blue Tang", "Clownfish", "Spotted Eagle Ray", "Longnose Butterfly Fish", "Moorish Idol", "Royal Gramma Fish"]
df = create_imagedataset_from_bing(class_names, 25, "data", test_split=0.2, license="Free to share and use")

# Fine-tune a ViT Model with the data (in 1-2 minutes on a GPU)
finetune_image_classifier(df[df["split"] == "train"], model_name="google/vit-base-patch16-224-in21k", output_model_folder="./model_folder", epochs=15)
df["prediction"], df["probs"], df["embeddings"] = generate_image_pred_probs_embeddings(   df["image"].values, model_name="./model_folder")

# Check the result and detect problematic clusters
spotlight.show(df, layout="https://spotlight.renumics.com/resources/image_classification_v1.0.json")