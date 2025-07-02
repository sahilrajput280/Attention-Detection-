# AIML Model Collection 🚀

This repository contains deep learning and computer vision model files, training scripts, and utility functions for AI/ML experimentation and deployment.

---

## 📂 Repository Structure

AIML/
├── DATA/ # Dataset folder (structure private or custom)
├── best_resnet50.pth # Trained ResNet50 model weights
├── yolov5s.pt # YOLOv5 small pre-trained weights
├── yolov8m.pt # YOLOv8 medium pre-trained weights
├── train_model.py # Script to train models on your dataset
├── test.py # Script to test trained models
├── save_photo.py # Utility to capture and save photos for datasets
├── predict_webcam.py # Predict objects in real-time using webcam
├── newpredict_webcam.py # Updated webcam prediction script

---

## Features

✅ **Object Detection**: YOLOv5 & YOLOv8 models for real-time detection tasks.  
✅ **Classification**: ResNet50 for image classification tasks.  
✅ **Training Pipelines**: Easily train on your dataset with `train_model.py`.  
✅ **Real-Time Webcam Prediction** for quick testing and demos.  
✅ **Dataset Utilities** for capturing and saving photos.

---

## Setup

1️⃣ **Clone this repository**:
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
  
2️⃣ Install dependencies:
pip install -r requirements.txt
(If requirements.txt is not present, let me know to generate one for your environment.)

3️⃣ Run training:
python train_model.py

4️⃣ Run webcam prediction:
python predict_webcam.py
