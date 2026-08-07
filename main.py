#!/usr/bin/env python3
"""
Human detection + tracking on a Raspberry Pi using Ultralytics YOLO26 nano (yolo26n).

Works with either a USB webcam (via OpenCV) or the Pi Camera Module
(via picamera2 — auto-detected if installed).

Uses an NCNN-exported model by default (much faster than .pt on a
Pi's ARM CPU). Export one once with:
    yolo export model=yolo26n.pt format=ncnn imgsz=416

Only the COCO 'person' class (id 0) is kept, so this runs as a
lightweight human detector. The horizontal position of the tracked
person relative to the center of the frame is converted into a
0-180 pan angle and streamed to an Arduino over USB serial, which
drives the pan motor (see head_obj_detectino.ino).

Serial protocol (to the Arduino):
    "A<angle>\n"  e.g. "A090\n" sets the target angle (0 = full left,
                  90 = center, 180 = full right).
    "e\n"         enable the motor (sent once at startup).
    "q\n"         disable the motor (sent on exit).

Usage:
    python3 main.py                          # default: webcam, show window
    python3 main.py --source picam           # use Pi Camera Module
    python3 main.py --no-display             # headless (no GUI window)
    python3 main.py --serial-port /dev/ttyACM0
    python3 main.py --no-serial              # detect/print only, don't send
"""

import argparse
import os
import time
import cv2
from ultralytics import YOLO

try:
    import serial
except ImportError:
    serial = None

# OpenCV's own thread pool can race with the ncnn backend's threads on the
# Pi's limited core count, corrupting the heap ("corrupted size vs. prev_size").
# Let ncnn manage threading instead.
cv2.setNumThreads(1)

PERSON_CLASS_ID = 0  # 'person' in the COCO dataset


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26n human tracking + Arduino pan control")
    p.add_argument("--model", default="yolo26n_ncnn_model",
                   help="Model weights: an NCNN export dir (default, fast on Pi CPU) "
                        "or a .pt file. Export with: "
                        "yolo export model=yolo26n.pt format=ncnn imgsz=416")
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
    p.add_argument("--serial-port", default="/dev/ttyACM0",
                   help="Serial port the Arduino is connected to.")
    p.add_argument("--baud", type=int, default=9600,
                   help="Serial baud rate (must match Serial.begin() on the Arduino).")
    p.add_argument("--no-serial", action="store_true",
                   help="Run detection only; don't open/send over serial.")
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


def open_arduino(port, baud):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # Arduino resets when the serial port opens; let it boot
    ser.reset_input_buffer()
    ser.write(b"e\n")  # enable the motor
    return ser


def send_angle(ser, angle):
    ser.write(f"A{angle:03d}\n".encode("ascii"))


def pick_target(result):
    """Choose which detected person to track: the one with the largest box (closest)."""
    boxes = result.boxes
    if len(boxes) == 0:
        return None
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    best_idx = int(areas.argmax())
    return boxes.xyxy[best_idx].tolist()  # [x1, y1, x2, y2]


def angle_from_position(center_x, frame_width):
    """Map a person's horizontal center position to a 0-180 pan angle.

    0 = person at the left edge, 90 = centered, 180 = person at the right edge.
    """
    ratio = max(0.0, min(1.0, center_x / frame_width))
    return round(ratio * 180)


def main():
    args = parse_args()

    if not args.no_serial and serial is None:
        raise RuntimeError(
            "pyserial is not installed. Run 'pip install pyserial' or pass --no-serial."
        )

    if not args.model.endswith(".pt") and not os.path.exists(args.model):
        raise FileNotFoundError(
            f"NCNN model dir '{args.model}' not found. Export it once with:\n"
            f"    yolo export model=yolo26n.pt format=ncnn imgsz={args.imgsz}"
        )

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Set up the camera source
    use_picam = args.source == "picam"
    if use_picam:
        cam = open_picam()
    else:
        cam = open_webcam(args.cam_index)

    ser = None
    if not args.no_serial:
        print(f"Opening Arduino on {args.serial_port} @ {args.baud} baud...")
        ser = open_arduino(args.serial_port, args.baud)

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

            frame_width = frame.shape[1]

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

            angle = None
            box = pick_target(result)
            if box is not None:
                x1, _, x2, _ = box
                center_x = (x1 + x2) / 2
                angle = angle_from_position(center_x, frame_width)
                if ser is not None:
                    send_angle(ser, angle)

            # FPS (exponential moving average for a stable readout)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps
            prev_t = now

            if angle is None:
                side = "no target"
            elif angle < 85:
                side = f"LEFT  (angle {angle})"
            elif angle > 95:
                side = f"RIGHT (angle {angle})"
            else:
                side = f"CENTER (angle {angle})"

            if args.no_serial:
                print(f"People: {count} | {side} | {fps:4.1f} FPS", end="\r", flush=True)

            if not args.no_display:
                annotated = result.plot()  # draws boxes + labels
                cv2.line(annotated, (frame_width // 2, 0), (frame_width // 2, frame.shape[0]),
                         (255, 0, 0), 1)
                cv2.putText(annotated, f"People: {count}  {side}  {fps:.1f} FPS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.imshow("YOLO26n Human Tracking", annotated)
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
        if ser is not None:
            ser.write(b"q\n")  # disable the motor
            ser.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()

