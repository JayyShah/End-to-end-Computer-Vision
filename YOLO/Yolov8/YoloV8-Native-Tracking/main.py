# Required Dependencies

import cv2
import supervision as sv
from ultralytics import YOLO

START = sv.Point(320, 0)
END = sv.Point(320, 480)

def main():
    model = YOLO("yolov8l.pt") # Downloading the model (Detection, segmentaton etc) 
    line_zone = sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    for result in model.track(source=0, show=True, stream=True):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int) 

        detections = detections[detections.class_id != 0] # Ignore a particular class 

        labels = [
            f"#{tracker_id}{model.model.names[class_id]} {confidence=0.2f}"
            for _,confidence, class_id, tracker_id
            in detections 
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)


        cv2.imshow("yolov8", frame)

        if (cv2.waitkey(30) == 27):
            break

if __name__ == "__main__":
    main()
