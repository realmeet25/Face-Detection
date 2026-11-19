# 👁️ Real-Time Multi-Cascade Face Detection & Eye-Privacy Anonymizer

A college minor project that detects faces in **images, videos, and live webcam streams** using multiple Haar Cascade classifiers, and can optionally **anonymize only the eyes** of each detected face — preserving the rest of the face while protecting identity.

Built with **pure OpenCV and NumPy** — no deep-learning frameworks required.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-required-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Keyboard Controls](#-keyboard-controls)
- [Output Files](#-output-files)
- [CSV Report Format](#-csv-report-format)
- [Configuration & Tuning](#-configuration--tuning)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Interactive Menu** | Console menu with Image, Video, Live Webcam, and Exit options |
| **Multi-Cascade Detection** | Combines Frontal (default), Frontal (alt2), Left Profile, and Right Profile cascades for better coverage of angled faces |
| **Non-Maximum Suppression** | Merges overlapping detections using IoU + containment checks to remove duplicate boxes |
| **Eye-Only Privacy Blur** | Detects eyes inside each face and blurs *only* the eye region (with a fallback eye-strip blur if eyes aren't detected, e.g., glasses or head tilt) |
| **CLAHE Preprocessing** | Normalizes lighting before detection without amplifying noise |
| **Automatic Face Cropping** | Saves every detected face as a separate image |
| **CSV Analytics Report** | Logs timestamp, source, mode, face count, and processing time for every run |
| **Live Webcam Mode** | Real-time FPS counter, snapshot capture, and a live blur toggle |

---

## 🧠 How It Works

```
Input (Image / Video / Webcam)
        │
        ▼
Grayscale + CLAHE (lighting normalization)
        │
        ▼
Multi-Cascade Detection
  ├─ Frontal (default)
  ├─ Frontal (alt2)
  ├─ Left profile
  └─ Right profile  (image flipped horizontally, boxes mapped back)
        │
        ▼
Non-Maximum Suppression (IoU > 0.3 or containment > 0.5)
        │
        ▼
Per-face processing
  ├─ Draw bounding box + label
  ├─ (Optional) Detect eyes in upper 55% of face → Gaussian blur
  └─ Crop & save face
        │
        ▼
Output image/video + CSV log
```

**Why multiple cascades?** A single frontal Haar cascade misses faces turned sideways. Running several cascades and merging results with NMS improves recall while keeping the output clean.

---

## 📁 Project Structure

```
.
├── face_detection.py        # Main script (rename to match your file)
├── requirements.txt
├── README.md
└── output/                  # Auto-created on first run
    ├── cropped_faces/       # Individual face crops
    ├── detection_report.csv # Analytics log
    └── ...                  # Annotated images, videos, snapshots
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy
```

> **Note:** Haar Cascade XML files ship with `opencv-python` (via `cv2.data.haarcascades`), so no extra downloads are needed.

---

## ▶️ Usage

Run the script:

```bash
python face_detection.py
```

You'll see the interactive menu:

```
====================================================
   FACE DETECTION & EYE-PRIVACY ANONYMIZER (v2.0)
====================================================
  [1] Detect Faces in Image
  [2] Detect Faces in Video
  [3] Live Webcam Face Detection (Real-Time FPS)
  [4] Exit
====================================================
```

### Option 1 — Image
1. Enter the path to an image (absolute, or relative to the script folder).
2. Choose whether to enable **Eye-Only Privacy Blur** (`y`/`n`).
3. View the original and processed images side by side. Press **Enter** in the console to return to the menu.

### Option 2 — Video
1. Enter the path to a video file.
2. Toggle eye-blur live with **B**; stop with **Q**.
3. The processed video is saved to `output/<name>_detected.mp4`.

### Option 3 — Live Webcam
1. Uses camera index `0`.
2. Shows real-time FPS, face count, and blur status.
3. Save snapshots with **S**.

---

## ⌨️ Keyboard Controls

> Click the OpenCV window first so it has keyboard focus.

| Key | Mode | Action |
|---|---|---|
| `B` | Video / Webcam | Toggle eye-only privacy blur |
| `S` | Webcam | Save snapshot + crop all faces |
| `Q` | Video / Webcam | Quit and return to menu |

---

## 📂 Output Files

| File | Location | Description |
|---|---|---|
| `<name>_detected.jpg` | `output/` | Image with bounding boxes |
| `<name>_eyes_blurred.jpg` | `output/` | Image with eye-only blur applied |
| `<name>_detected.mp4` | `output/` | Processed video |
| `webcam_snap_<timestamp>.jpg` | `output/` | Webcam snapshot |
| `<prefix>_face_<n>.jpg` | `output/cropped_faces/` | Individual cropped faces |
| `detection_report.csv` | `output/` | Run-by-run analytics log |

---

## 📊 CSV Report Format

Every run appends a row to `output/detection_report.csv`:

| Column | Description |
|---|---|
| `Timestamp` | Date and time of the run |
| `Source` | File name, or `Webcam_Live` |
| `Type` | `Image`, `Video`, or `Webcam` |
| `Privacy_Mode` | `Eye-Blur` or `Bounding-Box` |
| `Faces_Detected` | Face count (average per frame for video, peak for webcam) |
| `Processing_Time_ms` | Total processing time in milliseconds |

**Example:**

```csv
Timestamp,Source,Type,Privacy_Mode,Faces_Detected,Processing_Time_ms
2026-11-19 14:32:10,group_photo.jpg,Image,Eye-Blur,5,84.21
```

---

## 🔧 Configuration & Tuning

Key parameters live inside `detect_faces()` and `apply_eye_privacy_blur()`:

| Parameter | Where | Effect |
|---|---|---|
| `scaleFactor` (1.08) | `detect_faces` | Lower = more thorough but slower |
| `minNeighbors` (4–6) | `detect_faces` | Higher = fewer false positives |
| `min_dim` | `detect_faces` | Minimum face size (6% of the smaller image side, at least 35 px) |
| NMS thresholds | `detect_faces` | `iou > 0.3` or `containment > 0.5` suppresses duplicates |
| Eye search area | `apply_eye_privacy_blur` | Upper 55% of the face box |
| Blur kernel | `apply_eye_privacy_blur` | Scales with eye/face size |

---

## ⚠️ Limitations

- Haar Cascades are fast but less accurate than modern deep-learning detectors, especially with heavy occlusion, extreme angles, or poor lighting.
- Eye-blur is intended as a **visual privacy demo**, not a guaranteed anonymization method — Gaussian blur is not cryptographically irreversible.
- Webcam mode uses camera index `0` by default.
- Requires a display for preview windows (the script handles headless environments gracefully but won't show previews).

---

## 🚀 Future Improvements

- [ ] Add a DNN-based detector (e.g., OpenCV DNN / YuNet) for higher accuracy
- [ ] Pixelation and full-face blur options
- [ ] Command-line arguments (`argparse`) for batch processing
- [ ] Configurable camera index
- [ ] GUI front-end (Tkinter / Streamlit)
- [ ] Face tracking across video frames to reduce flicker

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** OpenCV (`opencv-python`), NumPy
- **Techniques:** Haar Cascade classifiers, CLAHE, Non-Maximum Suppression, Gaussian blur

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Meet**
B.Tech CSE (AI & Data Science), JECRC University, Jaipur

⭐ If you found this project useful, consider giving it a star!
