from inference.models.utils import get_roboflow_model
import cv2

# Roboflow model
model_name = "face-detection-mik1i"
model_version = "18"

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get Roboflow face model (this will fetch the model from Roboflow)
model = get_roboflow_model(
    model_id="{}/{}".format(model_name, model_version),
    #Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key="ROBOFLOW_API_KEY"
)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read successfully, display it
    if ret:
        # Run inference on the frame
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

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Could not read frame.")
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()