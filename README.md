# 🥔 Potato Disease Classifier API

A Machine Learning API built using **FastAPI** that detects **potato leaf diseases** from uploaded images. It also uses an **Autoencoder** to detect if the uploaded image is **not a potato leaf** (anomaly detection).

---

## 🧠 Model Capabilities

This project includes **two models**:
1. **Disease Classifier** (`1.h5` or `1.keras`)  
   Classifies valid potato leaf images into:
   - Early Blight
   - Late Blight
   - Healthy

2. **Anomaly Detector** (`autoencoder_256.keras`)  
   Detects whether an image is likely **not a potato leaf**, without needing non-potato images during training.

---

## 📦 Features

 Trained CNN model for disease classification  
 Autoencoder-based anomaly detection  
 `/predict` endpoint with:
- File upload
- Confidence score
- Reconstruction error
- Warning for invalid leaves

 RESTful API using **FastAPI**  
 Interactive docs via **Swagger UI**  
 CORS enabled for frontend integration

---

## 📁 Project Structure


 potato_disease/
├── main.py # FastAPI application
├── 1.h5 or 1.keras # Trained ML model
├── autoencoder_256.keras # Autoencoder model for anomaly detection
├── README.md # Project documentation
└── requirements.txt



## output 
![image](https://github.com/user-attachments/assets/41e4c86e-57e2-48d5-aea9-63ae49e10e7b)
![image](https://github.com/user-attachments/assets/338733d0-7a08-4e11-9f21-a00ed6c8d9bd)
<img width="985" height="261" alt="image" src="https://github.com/user-attachments/assets/150ec1ae-2197-48d6-951a-8d72cffbb9fd" />



