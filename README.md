# 🎯 Attention Detection - AIML Model Collection

This repository contains a collection of deep learning and computer vision models focused on attention detection using webcam input. It includes training scripts, prediction modules, and pre-trained models to support experimentation and deployment of AI/ML solutions.

---

## 📁 Project Structure

Attention-Detection-/
│
├── DATA/ # Dataset folder (not included)
│
├── best_resnet50.pth # Pre-trained ResNet50 model
├── yolov5s.pt # YOLOv5 small model weights
├── yolov8m.pt # YOLOv8 medium model weights
│
├── newpredict_webcam.py # New version of webcam prediction script
├── predict_webcam.py # Webcam attention prediction script
├── save_photo.py # Captures and saves webcam images
├── test.py # Test script for model validation
├── train_model.py # Model training script
│
├── README.md # Project documentation

---

## 🚀 Features

- Real-time attention detection using webcam
- Custom model training pipeline (ResNet, YOLO)
- Pre-trained models included
- Modular scripts for training, testing, and inference
- Easy integration with new data

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sahirajput280/Attention-Detection-.git
   cd Attention-Detection-


📷 Usage
1. Run Attention Detection (Webcam)
  python predict_webcam.py

2. Train Your Own Model
  python train_model.py

3. Test the Model
  python test.py

4. Save a Photo from Webcam
  python save_photo.py
