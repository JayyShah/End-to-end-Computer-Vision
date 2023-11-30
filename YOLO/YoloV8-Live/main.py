import cv2
from ultralytics import YOLO
import argparse
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YoloV8 Live")
    parser.add_argument(
        "--webcam-resolution",
        defaults=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    model = YOLO("yolov8l.pt")
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 1,
        text_scale = 0.5
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)

    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh = tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.color.red(), 
        thickness=2, 
        text_thickness=1, 
        text_scale=0.5
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id != 0] # Change the Class_id to a particular class to detect that category only
        labels = [
            f"{model.model.names{class_id}} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("yolov8", frame)

        if(cv2.waitkey(30) == 27):
            break

if __name__ == "__main__":
    main()