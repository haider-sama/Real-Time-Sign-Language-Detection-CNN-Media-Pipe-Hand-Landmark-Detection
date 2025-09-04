# Real-Time-Sign-Language-Detection-using-CNN-and-Media-Pipe-for-Hand-Landmark-Detection

## 📖 Introduction
This project implements a **Real-Time Sign Language Detection System** using **Convolutional Neural Networks (CNNs)** and **MediaPipe** for hand landmark detection.  

The system captures live video input, detects **hand landmarks**, and classifies gestures into **sign language alphabets (A–Z)** in real time.  
It demonstrates the integration of **deep learning** and **computer vision** for accessibility-focused applications.  

---

## 🔬 Methodology
### 1. **Hand Landmark Detection**
- Uses **MediaPipe** to detect and track **hand landmarks** from webcam input.  
- Normalizes and preprocesses landmarks into CNN-compatible features.  

### 2. **Feature Extraction**
- Extracts **joint positions** and **relative distances** between hand landmarks.  

### 3. **Gesture Classification**
- A **CNN model** processes features and predicts the **sign language alphabet**.  

### 4. **Real-Time Feedback**
- Predictions are displayed live on screen for **immediate user feedback**.  

---

## 🖥️ System Requirements
### Software
- Python 3.x  
- OpenCV  
- MediaPipe  
- PyTorch  
- Pandas  

### Hardware
- A computer with a **webcam**.  
- GPU (recommended) for faster model training & testing.  

---

## 🚀 How to Run
### 1. Install Dependencies
```bash
pip install -r requirements.txt

Or install manually:
```bash
pip install opencv-python mediapipe torch numpy pandas

### Run Real-Time Detection
```bash
python realTime_45.py

### Train the CNN Model (Optional)
```bash
python training.py

### Test the CNN Model (Optional)
```bash
python testCNN.py

## 📂 Project Structure

📁  Sign-Language-Detection/
<br />
└──📄 CNNModel.py                  # CNN architecture
└── 📄 CNNModel.ipynb               # Notebook for CNN
└── 📄 CNN_model_alphabet_SIBI.pth  # Pre-trained CNN weights
└── 📄 handLandMarks.py             # MediaPipe hand landmark detection
└── 📄 mediapipeHandDetection.py    # Real-time MediaPipe tracking
└── 📄 mediapipeHandDetection.ipynb # Notebook version of hand tracking
└── 📄 realTime_45.py               # Real-time detection script
└── 📄 realTime_45.ipynb            # Notebook version of real-time detection
└── 📄 training.py                  # CNN training script
└── 📄 training.ipynb               # Notebook version of training
└── 📄 testCNN.py                   # CNN testing script
└── 📄 testCNN.ipynb                # Notebook version of testing
└── 📄 Project_Report_02_ES_II_45.pdf # Full project report
└── 📄 README.md                    # Project documentation

📌 Key Insights

- CNN + MediaPipe integration enables robust real-time sign recognition.
- Achieves high accuracy on alphabet classification with optimized preprocessing.
- Provides a scalable base for gesture-based human-computer interaction (HCI) and accessibility solutions.
