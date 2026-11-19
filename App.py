"""
Real-Time Multi-Cascade Face Detection & Eye-Privacy Anonymizer
College Minor Project — Pure OpenCV & NumPy Implementation

Features:
  1. Interactive Menu Loop (Image, Video, Live Webcam, Exit)
  2. Multi-Cascade Face Detection (Frontal Default, Frontal Alt2, Left/Right Profile) + NMS
  3. Eye-Only Privacy Blur Toggle (Blurs only the eye region inside each detected face)
  4. Side-by-Side Comparison Window (Given vs. Output with green squares, IDLE-safe rendering)
  5. HD Live Webcam with [F] Fullscreen Toggle, [B] Eye-Blur, [S] Snapshot & Auto CSV Report
"""

import os
import csv
import time
from datetime import datetime
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Setup & Model Loading
# ---------------------------------------------------------------------------

print("OpenCV Version:", cv2.__version__)
assert hasattr(cv2, "CascadeClassifier"), (
    "OpenCV installation is incorrect. Reinstall with: pip install opencv-python"
)

# Load pre-trained Haar Cascade XML models (Face + Eyes)
HAAR_DIR = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")
face_cascade_alt = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_eye.xml")

if (
    face_cascade.empty()
    or face_cascade_alt.empty()
    or profile_cascade.empty()
    or eye_cascade.empty()
):
    raise RuntimeError("Unable to load one or more Haar Cascade XML models!")

print("All Face & Eye Detection Models Loaded Successfully!")

# Output directories & CSV report setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CROPPED_DIR = os.path.join(OUTPUT_DIR, "cropped_faces")
REPORT_PATH = os.path.join(OUTPUT_DIR, "detection_report.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

if not os.path.isfile(REPORT_PATH):
    with open(REPORT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Timestamp", "Source", "Type", "Privacy_Mode", "Faces_Detected", "Processing_Time_ms"]
        )


def log_report(
    source_name: str, media_type: str, blur_eyes: bool, face_count: float, elapsed_ms: float
) -> None:
    """Append detection metrics to detection_report.csv for project documentation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_label = "Eye-Blur" if blur_eyes else "Bounding-Box"
    with open(REPORT_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [timestamp, source_name, media_type, mode_label, face_count, f"{elapsed_ms:.2f}"]
        )


# ---------------------------------------------------------------------------
# 2. Display, Side-by-Side Canvas & Preprocessing Helpers
# ---------------------------------------------------------------------------

def make_side_by_side(
    left_img, right_img, left_title="Given Image", right_title="Output Image"
):
    """Combine Given and Output images into one clean side-by-side comparison canvas."""
    h, w = left_img.shape[:2]

    max_total_w = 1300
    if w * 2 > max_total_w:
        scale = max_total_w / (w * 2)
        new_w, new_h = int(w * scale), int(h * scale)
        left_img = cv2.resize(left_img, (new_w, new_h))
        right_img = cv2.resize(right_img, (new_w, new_h))
        h, w = new_h, new_w

    header_h = 40
    left_header = np.zeros((header_h, w, 3), dtype=np.uint8)
    right_header = np.zeros((header_h, w, 3), dtype=np.uint8)

    cv2.putText(
        left_header, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
    )
    cv2.putText(
        right_header, right_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
    )

    left_panel = np.vstack([left_header, left_img])
    right_panel = np.vstack([right_header, right_img])

    divider = np.full((h + header_h, 6, 3), 255, dtype=np.uint8)
    return np.hstack([left_panel, divider, right_panel])


def safe_imshow(window_name, img, pump_events=True, resizable=False):
    """Show an OpenCV window reliably in Windows / Python IDLE."""
    try:
        flag = cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(window_name, flag)
        cv2.imshow(window_name, img)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        if pump_events:
            for _ in range(5):
                cv2.waitKey(20)
        return True
    except cv2.error:
        return False


def preprocess_gray(bgr_img):
    """Convert to grayscale and apply CLAHE to normalize lighting without amplifying noise."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_faces(gray_img):
    """Run multi-cascade face detection and merge overlapping boxes using IoU + containment NMS."""
    all_boxes = []
    min_dim = max(35, int(min(gray_img.shape[:2]) * 0.06))
    min_face_size = (min_dim, min_dim)

    # Frontal (default)
    all_boxes.extend(
        face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.08, minNeighbors=5, minSize=min_face_size
        )
    )
    # Frontal (alt2)
    all_boxes.extend(
        face_cascade_alt.detectMultiScale(
            gray_img, scaleFactor=1.08, minNeighbors=4, minSize=min_face_size
        )
    )
    # Left profile
    all_boxes.extend(
        profile_cascade.detectMultiScale(
            gray_img, scaleFactor=1.08, minNeighbors=6, minSize=min_face_size
        )
    )
    # Right profile (horizontally flipped)
    flipped = cv2.flip(gray_img, 1)
    profiles_right = profile_cascade.detectMultiScale(
        flipped, scaleFactor=1.08, minNeighbors=6, minSize=min_face_size
    )
    w_img = gray_img.shape[1]
    for (x, y, w, h) in profiles_right:
        all_boxes.append((w_img - x - w, y, w, h))

    if len(all_boxes) == 0:
        return []

    # Non-Maximum Suppression (IoU + Containment)
    boxes = np.array(all_boxes, dtype=float)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = areas.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        iou = inter / (areas[i] + areas[rest] - inter)
        containment = inter / areas[rest]

        suppress = (iou > 0.3) | (containment > 0.5)
        order = rest[~suppress]

    return boxes[keep].astype(int)


def apply_eye_privacy_blur(output_img, gray_img, face_box):
    """
    Detect eyes inside the upper 55% of a detected face and blur ONLY the eye region.
    Falls back to a narrow horizontal eye strip if glasses or head angle hide the eyes.
    """
    x, y, w, h = face_box
    upper_h = max(1, int(h * 0.55))
    roi_gray = gray_img[y : y + upper_h, x : x + w]
    roi_color = output_img[y : y + upper_h, x : x + w]

    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(12, w // 10), max(12, h // 10)),
    )

    if len(eyes) > 0:
        for (ex, ey, ew, eh) in eyes:
            pad_x = int(ew * 0.15)
            pad_y = int(eh * 0.15)
            x1 = max(0, ex - pad_x)
            y1 = max(0, ey - pad_y)
            x2 = min(w, ex + ew + pad_x)
            y2 = min(upper_h, ey + eh + pad_y)

            eye_patch = roi_color[y1:y2, x1:x2]
            if eye_patch.size > 0:
                k = max(23, (ew // 2) | 1)
                roi_color[y1:y2, x1:x2] = cv2.GaussianBlur(eye_patch, (k, k), 30)
    else:
        strip_y1 = int(h * 0.22)
        strip_y2 = int(h * 0.48)
        strip_x1 = int(w * 0.12)
        strip_x2 = int(w * 0.88)
        eye_strip = output_img[y + strip_y1 : y + strip_y2, x + strip_x1 : x + strip_x2]
        if eye_strip.size > 0:
            k = max(25, (w // 3) | 1)
            output_img[y + strip_y1 : y + strip_y2, x + strip_x1 : x + strip_x2] = cv2.GaussianBlur(
                eye_strip, (k, k), 30
            )


def annotate_frame(frame, gray, faces, blur_eyes=False):
    """Draw green bounding squares/rectangles on faces and optionally apply eye-only blur."""
    for i, (x, y, w, h) in enumerate(faces):
        if blur_eyes:
            apply_eye_privacy_blur(frame, gray, (x, y, w, h))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Face {i + 1} [Eyes Blurred]" if blur_eyes else f"Face {i + 1}"
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )


def save_cropped_faces(original_img, faces, prefix):
    """Crop and save each detected face into output/cropped_faces/."""
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = original_img[y : y + h, x : x + w]
        if face_crop.size > 0:
            crop_name = f"{prefix}_face_{i + 1}.jpg"
            cv2.imwrite(os.path.join(CROPPED_DIR, crop_name), face_crop)


# ---------------------------------------------------------------------------
# 3. Mode 1: Image Processing
# ---------------------------------------------------------------------------

def process_image(image_path: str) -> None:
    if not os.path.isabs(image_path):
        image_path = os.path.join(SCRIPT_DIR, image_path)

    if not os.path.isfile(image_path):
        print(f"Error: File not found — {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image — {image_path}")
        return

    blur_choice = input("Enable Eye-Only Privacy Blur? (y/n) [default: n]: ").strip().lower()
    blur_eyes = blur_choice == "y"

    start_time = time.perf_counter()
    gray = preprocess_gray(image)
    faces = detect_faces(gray)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_cropped_faces(image, faces, base_name)

    result = image.copy()
    annotate_frame(result, gray, faces, blur_eyes=blur_eyes)

    suffix = "_eyes_blurred" if blur_eyes else "_detected"
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}{suffix}.jpg")
    cv2.imwrite(output_path, result)

    log_report(os.path.basename(image_path), "Image", blur_eyes, len(faces), elapsed_ms)

    print("\n================================")
    print("       FACE DETECTION RESULT")
    print("================================")
    print(f"Total Faces Detected : {len(faces)}")
    print(f"Privacy Mode         : {'Eye-Only Blur' if blur_eyes else 'Bounding Box'}")
    print(f"Processing Time      : {elapsed_ms:.2f} ms")
    print(f"Output Image Saved   : {output_path}")
    if len(faces) > 0:
        print(f"Cropped Faces Saved  : {CROPPED_DIR}")
    print(f"CSV Report Updated   : {REPORT_PATH}")

    mode_tag = "Eye-Blur Output" if blur_eyes else "Output Image"
    comparison = make_side_by_side(
        image,
        result,
        left_title="Given Image",
        right_title=f"{mode_tag} ({len(faces)} Faces)",
    )

    if safe_imshow("Face Detection — Given vs Output", comparison, pump_events=True):
        input("\nPress Enter in this console to close image and return to menu...")
        cv2.destroyAllWindows()
        cv2.waitKey(20)
    else:
        print("(No display available — output saved to file)")


# ---------------------------------------------------------------------------
# 4. Mode 2: Video File Processing (Side-by-Side + Live 'B' Eye-Blur Toggle)
# ---------------------------------------------------------------------------

def process_video(video_path: str) -> None:
    if not os.path.isabs(video_path):
        video_path = os.path.join(SCRIPT_DIR, video_path)

    if not os.path.isfile(video_path):
        print(f"Error: File not found — {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video — {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(OUTPUT_DIR, f"{base_name}_detected.mp4")
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    blur_eyes = False
    total_frames = 0
    total_detected = 0
    delay = max(1, int(1000 / fps))
    start_time = time.perf_counter()

    print(f"\nProcessing Video: {video_path}")
    print("Controls (click video window first): Press 'B' to toggle Eye-Blur | Press 'Q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        given_frame = frame.copy()
        gray = preprocess_gray(frame)
        faces = detect_faces(gray)

        annotate_frame(frame, gray, faces, blur_eyes=blur_eyes)

        mode_text = "ON (Eyes Only)" if blur_eyes else "OFF"
        cv2.putText(
            frame,
            f"Faces: {len(faces)} | Eye-Blur [B]: {mode_text}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

        writer.write(frame)
        total_frames += 1
        total_detected += len(faces)

        comparison_frame = make_side_by_side(
            given_frame,
            frame,
            left_title="Given Video",
            right_title=f"Output Video ({len(faces)} Faces)",
        )

        if safe_imshow("Video Face Detection — Given vs Output", comparison_frame, pump_events=False):
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                print("Stopped early by user.")
                break
            elif key == ord("b"):
                blur_eyes = not blur_eyes

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    cv2.waitKey(20)

    avg_faces = round(total_detected / max(1, total_frames), 2)
    log_report(os.path.basename(video_path), "Video", blur_eyes, avg_faces, elapsed_ms)

    print("\nVideo Processing Completed!")
    print(f"Total Frames Processed : {total_frames}")
    print(f"Average Faces / Frame  : {avg_faces}")
    print(f"Output Video Saved     : {output_video_path}")
    print(f"CSV Report Updated     : {REPORT_PATH}")


# ---------------------------------------------------------------------------
# 5. Mode 3: Live Webcam Detection (HD Resolution + [F] Fullscreen Toggle)
# ---------------------------------------------------------------------------

def process_webcam() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam (camera index 0).")
        return

    # Request 1280x720 (16:9 widescreen) instead of default 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_name = "Live Webcam — Face & Eye Privacy Detector"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1080, 620)  # Large default window size

    blur_eyes = False
    is_fullscreen = False
    prev_time = time.perf_counter()
    session_start = prev_time
    max_faces_seen = 0

    print("\nStarting Live Webcam Detection...")
    print("Controls (click webcam window first):")
    print("  [F] Toggle Fullscreen / Windowed")
    print("  [B] Toggle Eye-Only Privacy Blur")
    print("  [S] Save Snapshot + Crop Faces")
    print("  [Q] Quit Webcam Mode\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy()

        # Detect on a slightly scaled-down copy if HD for fast real-time FPS
        h, w = frame.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        if scale < 1.0:
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            small_gray = preprocess_gray(small_frame)
            small_faces = detect_faces(small_gray)
            faces = [(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)) for (x, y, fw, fh) in small_faces]
            gray = preprocess_gray(frame)
        else:
            gray = preprocess_gray(frame)
            faces = detect_faces(gray)

        max_faces_seen = max(max_faces_seen, len(faces))
        annotate_frame(frame, gray, faces, blur_eyes=blur_eyes)

        curr_time = time.perf_counter()
        fps = 1.0 / max(1e-5, (curr_time - prev_time))
        prev_time = curr_time

        mode_text = "ON" if blur_eyes else "OFF"
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Faces: {len(faces)} | Eye-Blur [B]: {mode_text}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Keys: [F] Fullscreen  [B] Eye-Blur  [S] Snapshot  [Q] Quit",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)
            if not is_fullscreen:
                cv2.resizeWindow(win_name, 1080, 620)
        elif key == ord("b"):
            blur_eyes = not blur_eyes
            print(f"Eye-Only Privacy Blur: {'ENABLED' if blur_eyes else 'DISABLED'}")
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(OUTPUT_DIR, f"webcam_snap_{ts}.jpg")
            cv2.imwrite(snap_path, frame)
            save_cropped_faces(clean_frame, faces, f"webcam_{ts}")
            print(f"Snapshot saved: {snap_path}")

    total_ms = (time.perf_counter() - session_start) * 1000
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(20)

    log_report("Webcam_Live", "Webcam", blur_eyes, max_faces_seen, total_ms)
    print(f"\nWebcam session ended. Peak faces detected: {max_faces_seen}. Logged to CSV.")


# ---------------------------------------------------------------------------
# 6. Main Interactive Menu Loop
# ---------------------------------------------------------------------------

def main():
    while True:
        print("\n====================================================")
        print("   FACE DETECTION & EYE-PRIVACY ANONYMIZER (v2.1)")
        print("====================================================")
        print("  [1] Detect Faces in Image")
        print("  [2] Detect Faces in Video")
        print("  [3] Live Webcam Face Detection (HD + Fullscreen)")
        print("  [4] Exit")
        print("====================================================")

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            image_path = input("Enter image path: ").strip().strip('"')
            if image_path:
                process_image(image_path)
        elif choice == "2":
            video_path = input("Enter video path: ").strip().strip('"')
            if video_path:
                process_video(video_path)
        elif choice == "3":
            process_webcam()
        elif choice == "4" or choice.lower() in ("exit", "quit", "q"):
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
