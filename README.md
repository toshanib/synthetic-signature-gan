# 🖊️ Synthetic Signature Generator using Vanilla GAN (Mobile + Docker Deployment)

## 📌 Project Overview
This project implements a Vanilla Generative Adversarial Network (GAN) to generate realistic synthetic handwritten signatures(CEDAR DATASET) and deploy them through a mobile application. The system is designed to augment limited signature datasets and support signature verification systems used in banking, academic institutions, and identity authentication workflows.

The project includes:
- Vanilla GAN for signature generation
- Signature verification experiment
- FastAPI backend (Dockerized)
- Flutter mobile app frontend
- End-to-end reproducible AI deployment pipeline

---

## 🎯 Project Objectives
1. Learn the distribution of real handwritten signatures.
2. Generate realistic synthetic signature images using a Vanilla GAN.
3. Augment training data for signature verification systems.
4. Evaluate verification performance (Accuracy, FAR, FRR).
5. Deploy a mobile app connected to a GAN inference API.

---

## 🧠 System Architecture

### Pipeline
Real Signature Dataset  
        ↓  
Data Preprocessing (Grayscale, Resize, Normalize)  
        ↓  
Vanilla GAN Training (Generator + Discriminator)  
        ↓  
Trained Generator (G_final.pth)  
        ↓  
FastAPI Backend (Inference API)  
        ↓  
Flutter Mobile App (Android Emulator / Device)  
        ↓  
Real-time Synthetic Signature Generation  

### Deployment Architecture
Flutter Mobile App (Client)  
        ↓ HTTP Request  
http://10.0.2.2:8000  
        ↓  
Docker Container (Backend)  
- FastAPI (Uvicorn Server)  
- PyTorch GAN Model  
- /generate API Endpoint  

This follows a client-server microservice architecture used in real AI production systems.

---

## 🧬 GAN Model Architecture

### Generator (G)
- Input: 100-dimensional latent vector (z ~ N(0,1))
- Dense layer → Reshape (4×4×256)
- ConvTranspose2D + BatchNorm + ReLU (Upsampling blocks)
- Final Layer: Conv2D (1 channel) + Tanh
- Output: 64×64 grayscale synthetic signature image

### Discriminator (D)
- Input: 64×64 grayscale signature image
- Conv2D + LeakyReLU (0.2)
- Downsampling layers
- Flatten + Dense(1) + Sigmoid
- Output: Real/Fake probability

### Loss Function
- Binary Cross Entropy (BCE)
- Classical Vanilla GAN objective

---

## 📊 Dataset Information
- Dataset Type: Handwritten Signature Images from CEDAR DATASET
- Users: 5 users (subset)
- Samples: ~20 genuine signatures per user
- Image Size: 64×64 grayscale
- Preprocessing:
  - Grayscale conversion
  - Cropping and resizing
  - Normalization to [-1, 1]

---

## ⚙️ Training Configuration
- Optimizer: Adam
- Learning Rate: 0.0002
- Beta1: 0.5
- Epochs: 100–200
- Batch Size: 16
- Latent Dimension: 100
- Device: CPU (GPU optional)

---

## 🧪 Verification Module (Core Experiment)
A signature verification classifier is trained to distinguish:
- Genuine signatures
- Forged signatures

Evaluation Metrics:
- Accuracy
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)

Purpose:
To evaluate whether GAN-generated synthetic signatures improve verification robustness when real data is limited.

---

## 🚀 QUICK START (FULL SETUP GUIDE)

### 1️⃣ Clone the Repository
git clone https://github.com/toshanib/synthetic-signature-gan.git
cd synthetic-signature-gan

---

## 🐳 Option A — Recommended: Docker Backend (One Command Deployment)

### Step 1: Build Docker Image
docker build -t signature-gan .

### Step 2: Run Backend Container
docker run -p 8000:8000 signature-gan

Backend will run at:
http://localhost:8000

Open Swagger API:
http://localhost:8000/docs

Test generation endpoint:
http://localhost:8000/generate?n=8

---

## 💻 Option B — Manual Backend Setup (Without Docker)

### Step 1: Create Virtual Environment (Python 3.11 or 3.12)
python -m venv .venv

Activate on Windows:
.venv\Scripts\activate

### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Run FastAPI Backend
uvicorn backend.main:app --reload

Backend URL:
http://127.0.0.1:8000

Swagger Docs:
http://127.0.0.1:8000/docs

---

## 📱 Running the Flutter Mobile App

### Prerequisites
- Flutter SDK installed
- Android Studio / Emulator
- VS Code or Android Studio

### Step 1: Navigate to Mobile App Folder
cd mobile_app

### Step 2: Install Flutter Dependencies
flutter pub get

### Step 3: Run the App
flutter run

IMPORTANT:
For Android Emulator, the API base URL must be:
http://10.0.2.2:8000
NOT localhost.

---

## 🔌 API Endpoints
GET /generate?n=16  
Generates N synthetic signature images using the trained GAN.

GET /docs  
Swagger API documentation for testing endpoints.

---

## 📸 Features
- Real-time synthetic signature generation
- Dockerized AI backend (FastAPI + PyTorch)
- Flutter mobile application UI
- Adjustable number of generated signatures
- GAN architecture information screen
- Verification experiment integration
- Reproducible deployment environment

---

## 🐳 Why Docker is Used
The backend is containerized to ensure:
- Reproducibility across systems
- Easy setup with one command
- Consistent dependency management
- Production-style AI deployment

This follows modern MLOps and AI system design practices.

---

## Results
Epoch 20 Generated Sample: 
<img width="393" height="396" alt="image" src="https://github.com/user-attachments/assets/446b85db-41f4-41c3-bec0-eef23e8f5751" />

Epoch 200 Generated Sample: 

<img width="391" height="396" alt="image" src="https://github.com/user-attachments/assets/c73f2780-feca-4019-afac-875906b77041" />

Swagger API/FastAPI Results: 

<img width="1477" height="1352" alt="image" src="https://github.com/user-attachments/assets/69ef7f14-f606-45fb-af92-e942b2f4bca6" />

Android Mobile App Emulator: 

<img width="621" height="1402" alt="image" src="https://github.com/user-attachments/assets/a71e97d1-982d-44c6-b5d6-09ffd00df173" />


--- Evaluation Results ---
FID Proxy Score      : 0.0001
Diversity Score      : 5.7626
Real Mean Pixel      : 0.9194
Fake Mean Pixel      : 0.9106

Test samples: 24

===== Verification Evaluation Results =====
Accuracy                : 0.9583
False Acceptance Rate   : 0.0455
False Rejection Rate    : 0.0000
TP: 2, TN: 21, FP: 1, FN: 0

---

## 🔮 Future Work
- Conditional GAN (user-specific signatures)
- Higher resolution signatures (128×128 or 256×256)
- Siamese Network for advanced signature verification
- Cloud deployment (AWS / Streamlit / Firebase)
- Real-time model monitoring and versioning

---

## 🏁 Conclusion
This project demonstrates a complete end-to-end AI system for synthetic signature generation using a Vanilla GAN, integrated with a Dockerized FastAPI backend and a Flutter mobile frontend. The system successfully augments limited signature datasets and supports robust signature verification workflows for real-world applications such as banking, academic authentication, and identity verification systems.
