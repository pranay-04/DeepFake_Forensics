# DeepFake_Forensics

## Overview

**DeepFake_Forensics** is an AI-powered toolkit for detecting deepfake images. Deepfakes are manipulated media created by machine learning algorithms to mimic real people, presenting risks in security and authenticity. This repository provides a robust pipeline—using modern neural networks and explainability tools—to identify and visualize deepfake faces.

---

## Table of Contents

- [1. Project Purpose](#1-project-purpose)
- [2. Directory and Important Files](#2-directory-and-important-files)
- [3. Installation and Setup](#3-installation-and-setup)
- [4. Dependencies and Module Rationale](#4-dependencies-and-module-rationale)
- [5. Workflow: Step-by-Step](#5-workflow-step-by-step)
- [6. Model and Pipeline Details](#6-model-and-pipeline-details)
- [7. Running the Project](#7-running-the-project)
- [8. Example Usage](#8-example-usage)
- [9. Data and Training](#9-data-and-training)
- [10. Explainability](#10-explainability)
- [11. Limitations & Extensions](#11-limitations--extensions)
- [12. Project Structure Example](#12-project-structure-example)
- [13. References](#13-references)
- [14. License and Citation](#14-license-and-citation)
- [15. Author and Contact](#15-author-and-contact)

---

## 1. Project Purpose

- Classify uploaded face images as real or deepfake ("fake") using deep learning.
- Provide confidence scores and visual explanations (GradCAM) for each prediction.
- Enable easy interaction and extension for research, forensics, and practical applications.

---

## 2. Directory and Important Files

The repository contains:

- `Deepfake_detection.ipynb` – Main notebook, combines model loading, inference, and UI.
- `resnetinceptionv1_epoch_32.pth` – Pre-trained model checkpoint.
- `requirements.txt` – Python dependencies to recreate the environment.
- Utility scripts (optional): data prep, training pipeline, additional visualizations.
- `README.md` – Project documentation (this file).
- `LICENSE` – License information.
- `data/` – Recommended directory for storing evaluation/training images (not supplied).

---

## 3. Installation and Setup

### Clone the repository
```bash
git clone https://github.com/Srinivas-18/DeepFake_Forensics.git
cd DeepFake_Forensics
```

### Install required Python libraries
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install gradio torch facenet-pytorch numpy pillow opencv-python pytorch-grad-cam
```

### Download the model checkpoint
Ensure `resnetinceptionv1_epoch_32.pth` is in the root directory. (Contact author if not supplied.)

---

## 4. Dependencies and Module Rationale

- **gradio**  
  _Why:_ Launches an interactive web app, making it easy to test images and view results.
- **torch (PyTorch)**  
  _Why:_ Main machine learning framework for model definition, training, and inference.
- **facenet_pytorch**  
  _Why:_ Supplies `MTCNN` for state-of-the-art face detection; `InceptionResnetV1` for facial feature extraction.
- **numpy, PIL (Python Imaging Library)**  
  _Why:_ Efficient image processing and array manipulations.
- **opencv-python (cv2)**  
  _Why:_ General image processing, overlays, blending (used for visualization).
- **pytorch-grad-cam**  
  _Why:_ Generates visual explanations ("attention maps") indicating regions important for model decisions.

All libraries are chosen for their reliability, active support, and state-of-the-art implementation.

---

## 5. Workflow: Step-by-Step

1. **Face Detection:**  
   Input image is scanned using `MTCNN` to extract face region.
2. **Preprocessing:**  
   Face is resized, normalized, and formatted for model input.
3. **Classification:**  
   The preprocessed face is fed to `InceptionResnetV1`, which computes whether the face is "real" or "fake."
4. **Prediction:**  
   The output is a _confidence score_; below 0.5 is "real," above 0.5 is "fake."
5. **Explainability:**  
   `GradCAM` generates a heatmap over the face, showing what regions influenced the prediction most.
6. **Visualization & Output:**  
   Gradio displays the label, confidence, and explanation overlay in the browser.

---

## 6. Model and Pipeline Details

- **InceptionResnetV1**:  
  Pretrained on VGGFace2 for facial features; re-trained for binary classification (`real` vs `fake`).
- **Face Detection (MTCNN)**:  
  Robust detection of faces in varied image conditions.
- **GradCAM**:  
  Visualizes the learned focus areas of the model for each decision.

**Checkpoint loading example:**
```python
model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)
checkpoint = torch.load('resnetinceptionv1_epoch_32.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
```

---

## 7. Running the Project

**Using Jupyter Notebook:**
1. Open `Deepfake_detection.ipynb` in Jupyter Notebook or Lab.
2. Run cells in order.
3. Gradio will launch an interface in your browser.

**Using Python script (if ported):**
1. Run the main script (not provided in repo at present).

**Input:**  
Upload an image via the Gradio app.

**Output:**  
- Prediction label (`real` or `fake`)
- Confidence scores (`real`, `fake`)
- Explanatory overlay image

---

## 8. Example Usage

**Notebook example:**
```python
interface = gr.Interface(
    fn=predict,
    inputs=[gr.inputs.Image(label="Input Image", type="pil")],
    outputs=[
        gr.outputs.Label(label="Class"),
        gr.outputs.Image(label="Face with Explainability", type="pil")
    ],
).launch()
```

Follow instructions in your browser after launching.

---

## 9. Data and Training

**Data:**
- The model is trained on real and deepfake faces.
- Recommended datasets: [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://celebdf.github.io/).

**Training (not provided):**
- Preprocess faces, balance classes.
- Fine-tune InceptionResnetV1 with binary cross-entropy loss.

---

## 10. Explainability

- Uses GradCAM to generate attention heatmaps.
- Visual overlay blends face with attention, making forensic analysis easier.
- Crucial for forensic and trust-focused applications.

---

## 11. Limitations & Extensions

- **Limitations:**
  - Targets static facial images only (no video yet).
  - Model accuracy depends on quality/diversity of training data.
  - Limited to faces detectable by MTCNN.

- **Potential Extensions:**
  - Video deepfake detection pipeline.
  - Handling multiple faces per image.
  - Adding REST API.
  - Training scripts for custom datasets.
  - Compare multiple deepfake detection models.

---

## 12. Project Structure Example

```
DeepFake_Forensics/
├── Deepfake_detection.ipynb            # Inference and UI notebook
├── resnetinceptionv1_epoch_32.pth      # Model weights
├── requirements.txt                    # Environment spec
├── README.md
├── LICENSE
├── data/                               # (Images for evaluation/training)
└── utils/                              # (Extra scripts/utilities)
```

---

## 13. References

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://celebdf.github.io/)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

## 14. License and Citation

This project is released under the [MIT License](LICENSE).

If you use this work in research, please cite the repository or author.

---

## 15. Author and Contact

Developed by [pranay-04](https://github.com/pranay-04).

For questions, issues, or improvements, please open an issue or pull request on GitHub.

---

**Thank you for using DeepFake_Forensics!**
