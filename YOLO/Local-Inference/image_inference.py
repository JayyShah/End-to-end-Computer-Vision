from inference.models.utils import get_roboflow_model
import cv2

# Image path
image_path = "photo.jpg"

# Roboflow model
model_name = "face-detection-mik1i"
model_version = "18"

# Get Roboflow face model (this will fetch the model from Roboflow)
model = get_roboflow_model(
    model_id="{}/{}".format(model_name, model_version),
    #Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key="ROBOFLOW_API_KEY"
)

# Load image with opencv
frame = cv2.imread(image_path)

# Inference image to find faces
results = model.infer(image=frame,
                        confidence=0.5,
                        iou_threshold=0.5)


# Plot image with face bounding box (using opencv)
if results[0]:
    bounding_box = results[0][0]
    print(bounding_box)

    x0, y0, x1, y1 = map(int, bounding_box[:4])
    
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255,255,0), 10)
    cv2.putText(frame, "Face", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# Show image
cv2.imshow('Image Frame', frame)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image