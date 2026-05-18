# ISL Letter Translator

A real-time Indian Sign Language (ISL) Letter Translator built using Python, OpenCV, MediaPipe, and TensorFlow.

---

## Features

* Real-time webcam hand tracking
* Single-hand ISL letter recognition
* Deep learning prediction using TensorFlow
* Confidence score display
* MediaPipe hand landmark detection

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy

---

## Dataset

Dataset used for training:

https://www.kaggle.com/datasets/kabilan03/sign-language-dataset

---

## Installation

### Clone Repository

```bash id="x2d0h0"
git clone https://github.com/YOUR_USERNAME/ISL-Letter-Translator.git
cd ISL-Letter-Translator
```

---

### Create Virtual Environment

```bash id="xj2xv3"
python -m venv .venv
```

---

### Activate Virtual Environment

#### Windows

```bash id="ij47m7"
.\.venv\Scripts\activate
```

---

### Install Required Packages

```bash id="08wgm5"
pip install -r requirements.txt
```

---

## Run the Project

```bash id="6q2r7o"
python app.py
```

Press `q` to exit the application.

---

## Required Files

Ensure these files are present inside the project folder:

```text id="4q6k99"
isl_letter_model_onehand.h5
classesOneHand.npy
```


---

## Author

Hemil
