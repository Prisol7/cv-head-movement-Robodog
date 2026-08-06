#!/usr/bin/env python3
"""
Human detection on a Raspberry Pi using Ultralytics YOLO26 nano (yolo26n).

Works with either a USB webcam (via OpenCV) or the Pi Camera Module
(via picamera2 — auto-detected if installed).

Only the COCO 'person' class (id 0) is kept, so this runs as a
lightweight human detector.

Usage:
    python3 human_detect.py                 # default: webcam, show window
    python3 human_detect.py --source picam  # use Pi Camera Module
    python3 human_detect.py --no-display     # headless (no GUI window)
    python3 human_detect.py --conf 0.4       # confidence threshold
    python3 human_detect.py --imgsz 320      # smaller = faster on Pi
"""

import argparse
import time
import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # 'person' in the COCO dataset


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26n human detection for Raspberry Pi")
    p.add_argument("--model", default="yolo26n.pt",
                   help="Model weights. Use a .pt file, or an NCNN export dir for speed.")
    p.add_argument("--source", default="webcam", choices=["webcam", "picam"],
                   help="Camera source: 'webcam' (USB/OpenCV) or 'picam' (Pi Camera).")
    p.add_argument("--cam-index", type=int, default=0,
                   help="OpenCV camera index (webcam source only).")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence threshold (0-1).")
    p.add_argument("--imgsz", type=int, default=416,
                   help="Inference image size. Lower = faster, less accurate.")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless: no GUI window (good for SSH / no monitor).")
    return p.parse_args()


def open_webcam(index, width=640, height=480):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {index}")
    return cap


def open_picam(width=640, height=480):
    # Imported here so the script still runs on non-Pi machines with a USB cam.
    from picamera2 import Picamera2
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1)  # let the sensor warm up
    return picam


def read_picam_frame(picam):
    # picamera2 gives RGB; OpenCV/YOLO drawing below expects BGR.
    frame_rgb = picam.capture_array()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Set up the camera source
    use_picam = args.source == "picam"
    if use_picam:
        cam = open_picam()
    else:
        cam = open_webcam(args.cam_index)

    print("Running. Press 'q' in the window (or Ctrl+C in terminal) to quit.")
    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            # Grab a frame
            if use_picam:
                frame = read_picam_frame(cam)
            else:
                ok, frame = cam.read()
                if not ok:
                    print("Failed to read frame; stopping.")
                    break

            # Inference: only look for people (classes=[0])
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
            result = results[0]
            count = len(result.boxes)

            # FPS (exponential moving average for a stable readout)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps
            prev_t = now

            if args.no_display:
                # Headless: just print. Replace this with your own logic
                # (GPIO, logging, alerts, etc.)
                print(f"People: {count} | {fps:4.1f} FPS", end="\r", flush=True)
            else:
                annotated = result.plot()  # draws boxes + labels
                cv2.putText(annotated, f"People: {count}  {fps:.1f} FPS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.imshow("YOLO26n Human Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if use_picam:
            cam.stop()
        else:
            cam.release()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
