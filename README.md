# Plate & SIM Detection (Flask + OCR + YOLO)

This project is a **Flask web application** that allows users to:
- Detect **vehicle license plates** using a YOLO model for detection and recognition.
- Extract **phone numbers from SIM cards** using EasyOCR.
- Support both **image uploads** and **real-time webcam streams**.
- Export annotated images with detected results.

---

## 📌 Features
- 🔍 Upload images of **cars** → Extract license plate numbers.  
- 📱 Upload images of **SIM cards** → Extract phone numbers.  
- 🎥 Use **real-time camera feed** for live detection.  
- 📊 Results displayed in a clean web interface.  

---

## 🛠️ Technologies Used
- **Python 3.10+**
- **Flask** (Web framework)
- **YOLOv8 (Ultralytics)** for object detection
- **EasyOCR** for text recognition
- **OpenCV** for image processing
- **Bootstrap** for front-end styling
