# Potato_Disease
# 🥔 Potato Disease Classifier API using FastAPI

This project is a machine learning web API built using **FastAPI** that detects potato leaf diseases from uploaded images. It classifies leaves into:

- **Early Blight**
- **Late Blight**
- **Healthy**

It uses a TensorFlow/Keras model exported as a `.h5` or `.keras` file and is deployed locally using FastAPI with an image upload endpoint.

---

## 📦 Features

- Image classification using a trained CNN model
- REST API with FastAPI
- File upload endpoint (`/predict`)
- Swagger UI for testing
- JSON response with predicted class and confidence
- CORS enabled (for frontend integration)

---

## 🧠 Model Details

- Developed using TensorFlow/Keras
- Trained on potato leaf disease dataset
- Exported as `1.h5` or `1.keras`

---

## 📁 Project Structure
 potato_disease/
├── main.py # FastAPI application
├── 1.h5 or 1.keras # Trained ML model
├── README.md # Project documentation
└── requirements.txt



## output 
![image](https://github.com/user-attachments/assets/41e4c86e-57e2-48d5-aea9-63ae49e10e7b)
![image](https://github.com/user-attachments/assets/338733d0-7a08-4e11-9f21-a00ed6c8d9bd)


