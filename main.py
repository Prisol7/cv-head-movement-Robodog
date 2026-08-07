#!/usr/bin/env python3
"""
Human detection + tracking on a Raspberry Pi using Ultralytics YOLO26 nano (yolo26n).

Works with either a USB webcam (via OpenCV) or the Pi Camera Module
(via picamera2 — auto-detected if installed).

Defaults to the .pt weights. An NCNN export can be faster on a Pi's
ARM CPU, but the ncnn backend has been observed to crash with a
glibc heap-corruption abort ("corrupted size vs. prev_size") on some
Pi setups due to thread-pool conflicts with OpenCV. Try it with
--model yolo26n_ncnn_model (export via
`yolo export model=yolo26n.pt format=ncnn imgsz=416`) and fall back
to .pt if it aborts.

Only the COCO 'person' class (id 0) is kept, so this runs as a
lightweight human detector. A sticky tracker locks onto one person
(matched frame-to-frame by proximity, tolerating a few missed frames)
so the servo doesn't jump between people, and their horizontal
position is converted into a 0-180 pan angle.

That angle is sent to the Arduino the same way head_obj_detectino.ino
expects it: as an 8-bit value bit-banged across 8 Pi GPIO output pins,
wired straight to the Arduino's assembleAng() input pins. Serial
(USB or GPIO UART) is used only for the one-off 'e' (enable) command
at startup and 'q' (disable) command at exit — the Arduino sketch's
own manual command interface, unchanged.

Wiring (Pi GPIO -> Arduino, BCM numbering, bit0..bit7 low-to-high):
    --gpio-pins bit  Pi BCM pin   Arduino pin   assembleAng() weight
    0                5            A0 (n)        1
    1                6            A1 (u)        2
    2                13           A2 (m)        4
    3                19           A3 (p)        8
    4                26           A4 (a)        16
    5                21           A5 (r)        32
    6                20           6   (t)       64
    7                16           7   (s)        128
    + a shared GND between the Pi and Arduino.
Override the pin list with --gpio-pins if wired differently.

Usage:
    python3 main.py                          # default: webcam, show window
    python3 main.py --source picam           # use Pi Camera Module
    python3 main.py --no-display             # headless (no GUI window)
    python3 main.py --serial-port /dev/ttyACM0   # 'e'/'q' commands over USB
    python3 main.py --no-serial              # skip the enable/disable commands
    python3 main.py --no-gpio                # detect/print only, don't drive pins
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

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None

# OpenCV's own thread pool can race with the ncnn backend's threads on the
# Pi's limited core count, corrupting the heap ("corrupted size vs. prev_size").
# Let ncnn manage threading instead.
cv2.setNumThreads(1)

PERSON_CLASS_ID = 0  # 'person' in the COCO dataset

DEFAULT_GPIO_PINS = [5, 6, 13, 19, 26, 21, 20, 16]  # bit0 (weight 1) .. bit7 (weight 128)


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26n human tracking + Arduino pan control")
    p.add_argument("--model", default="yolo26n.pt",
                   help="Model weights: a .pt file (default) or an NCNN export dir "
                        "for speed (can crash on some Pi setups — see module docstring). "
                        "Export with: yolo export model=yolo26n.pt format=ncnn imgsz=416")
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
    p.add_argument("--gpio-pins", default=",".join(str(p) for p in DEFAULT_GPIO_PINS),
                   help="Comma-separated BCM pin numbers for the 8-bit angle bus, "
                        "bit0 (weight 1) first. Must match the Arduino's "
                        "n,u,m,p,a,r,t,s wiring order. See module docstring.")
    p.add_argument("--no-gpio", action="store_true",
                   help="Run detection only; don't drive the GPIO angle pins "
                        "(e.g. for testing off a Pi).")
    p.add_argument("--serial-port", default="/dev/ttyACM0",
                   help="Serial port for the Arduino's 'e'/'q' enable-disable "
                        "commands only (not used for angle data). Use "
                        "/dev/serial0 if wired via the Pi's GPIO UART instead of USB.")
    p.add_argument("--baud", type=int, default=9600,
                   help="Serial baud rate (must match Serial.begin() on the Arduino).")
    p.add_argument("--no-serial", action="store_true",
                   help="Don't send the 'e'/'q' enable-disable commands over serial.")
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
    """Open the control-command serial link and enable the motor ('e')."""
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # let the Arduino finish booting/resetting
    ser.reset_input_buffer()
    ser.write(b"e\n")
    return ser


def setup_gpio(pins):
    GPIO.setmode(GPIO.BCM)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)


def send_angle_gpio(pins, angle):
    """Bit-bang a 0-180 angle across `pins`, matching the Arduino's assembleAng()."""
    angle = max(0, min(180, int(angle)))
    for bit, pin in enumerate(pins):
        GPIO.output(pin, (angle >> bit) & 1)
    print(f"Sending angle: {angle:3d} ({angle:08b})", end="\r", flush=True)


def get_person_boxes(result):
    """All detected person boxes this frame, as a list of (x1, y1, x2, y2)."""
    return result.boxes.xyxy.tolist()


def angle_from_position(center_x, frame_width):
    """Map a person's horizontal center position to a 0-180 pan angle.

    0 = person at the left edge, 90 = centered, 180 = person at the right edge.
    """
    ratio = max(0.0, min(1.0, center_x / frame_width))
    return round(ratio * 180)


class PersonTracker:
    """Sticky nearest-centroid tracker.

    Mirrors the LD19 lidar tracker's locked-target logic: once locked onto a
    person, keep following that same one (matched by proximity) instead of
    re-picking whichever box is currently largest every frame, and tolerate a
    few frames without a match before giving up the lock.
    """

    LOST_FRAMES_LIMIT = 8
    MATCH_FRACTION = 0.20  # of frame width

    def __init__(self):
        self.locked_x = None
        self.lost_frames = 0

    def update(self, boxes, frame_width):
        """boxes: list of (x1, y1, x2, y2) person boxes this frame.
        Returns a 0-180 servo angle, or None if there's nothing to track."""
        if not boxes:
            return self._handle_lost(frame_width)

        centers = [((x1 + x2) / 2, (x2 - x1) * (y2 - y1)) for x1, y1, x2, y2 in boxes]

        if self.locked_x is not None:
            cx, _ = min(centers, key=lambda c: abs(c[0] - self.locked_x))
            if abs(cx - self.locked_x) <= frame_width * self.MATCH_FRACTION:
                self.locked_x = cx
                self.lost_frames = 0
                return angle_from_position(self.locked_x, frame_width)
            return self._handle_lost(frame_width)

        # No lock yet: acquire the closest (largest-box) person.
        cx, _ = max(centers, key=lambda c: c[1])
        self.locked_x = cx
        self.lost_frames = 0
        return angle_from_position(self.locked_x, frame_width)

    def _handle_lost(self, frame_width):
        if self.locked_x is None:
            return None
        self.lost_frames += 1
        if self.lost_frames <= self.LOST_FRAMES_LIMIT:
            return angle_from_position(self.locked_x, frame_width)
        self.locked_x = None
        self.lost_frames = 0
        return None


def main():
    args = parse_args()

    if not args.no_serial and serial is None:
        raise RuntimeError(
            "pyserial is not installed. Run 'pip install pyserial' or pass --no-serial."
        )

    if not args.no_gpio and GPIO is None:
        raise RuntimeError(
            "RPi.GPIO is not installed/available. Run 'pip install RPi.GPIO' "
            "on the Pi, or pass --no-gpio to run detection only."
        )

    gpio_pins = [int(x) for x in args.gpio_pins.split(",")]
    if len(gpio_pins) != 8:
        raise ValueError(f"--gpio-pins needs exactly 8 pins, got {len(gpio_pins)}")

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

    if not args.no_gpio:
        setup_gpio(gpio_pins)

    ser = None
    if not args.no_serial:
        print(f"Opening Arduino control link on {args.serial_port} @ {args.baud} baud...")
        ser = open_arduino(args.serial_port, args.baud)

    tracker = PersonTracker()

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

            boxes = get_person_boxes(result)
            angle = tracker.update(boxes, frame_width)
            if angle is not None and not args.no_gpio:
                send_angle_gpio(gpio_pins, angle)

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

            if args.no_gpio:
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
        if not args.no_gpio:
            GPIO.cleanup()
        print("\nStopped.")


if __name__ == "__main__":
    main()

