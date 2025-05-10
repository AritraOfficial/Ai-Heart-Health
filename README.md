# AI Heart Health ❤️

An AI-powered web application for predicting the risk of heart disease based on patient data. This tool helps users assess heart health instantly using a deep learning model and an intuitive Streamlit interface.

---
## View 
![AIHEALTH](https://github.com/user-attachments/assets/666fa4f3-3cd2-4716-a613-425355a25ab8)

## 🧠 Project Overview

**AI Heart Health** is a machine learning-based web application designed to predict whether an individual is at risk of heart disease. It leverages a deep neural network trained on clinical data and presents the results through an interactive and user-friendly Streamlit interface.

This project aims to promote awareness, support early diagnosis, and assist in medical decision-making through AI.

---

## 🎯 Project Goals

- Predict the likelihood of heart disease using patient vitals and symptoms.
- Develop a user-centric and accessible web interface.
- Ensure accuracy, scalability, and usability in real-world scenarios.

---

## 📊 Dataset Overview

The model is trained on a structured dataset of patient records, including clinical and diagnostic information.

### **Key Features Used:**
- **Age** (years)
- **Sex** (Male/Female)
- **Chest Pain Type** (Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic)
- **Resting Blood Pressure** (mm Hg)
- **Cholesterol** (mg/dL)
- **Fasting Blood Sugar** (>120 mg/dL)
- **Resting ECG Results**
- **Maximum Heart Rate Achieved**
- **Exercise-Induced Angina**
- **ST Depression (Oldpeak)**
- **ST Segment Slope**

### **Target Variable:**
- `0`: Healthy
- `1`: Heart Disease

---

## 🛠️ Tech Stack

### ✅ Languages & Libraries:
- **Python** – Core programming language
- **Pandas, NumPy** – Data handling and preprocessing
- **Matplotlib, Seaborn** – Exploratory Data Analysis (EDA)
- **Scikit-learn** – Scaling, train/test split, evaluation metrics
- **TensorFlow / Keras** – Deep learning model development
- **Pickle, JSON** – Saving scalers and feature info
- **Streamlit** – UI and deployment

---

## 🧩 Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
````

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Output**: Probability of heart disease

---

## 🌐 Streamlit Web Application

The app allows users to:

* Input patient health data via the sidebar
* Instantly receive prediction results and health suggestions
* View confidence level of prediction

### 👉 Run the App Locally

```bash
# Clone the repository
git clone https://github.com/AritraOfficial/Ai-Heart-Health.git
cd Ai-Heart-Health

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Repository Structure

```
ai-heart-health/
├── app.py                 # Streamlit UI for prediction
├── main.ipynb             # Model training and preprocessing logic
├── heart_disease_model.h5 # Saved Keras model
├── scaler.pkl             # Saved Scikit-learn scaler
├── feature_names.json     # JSON file containing input feature names
├── dataset.csv            # dataset
├── Detect Heart Disease.pdf  # Dataset and column descriptions
└── README.md              # Project documentation
```

---

## 📌 Disclaimer

This application is intended for educational and preliminary screening purposes only. It is **not a substitute for professional medical advice**. Always consult a qualified healthcare provider for any diagnosis or treatment.

---

## 🙌 Acknowledgments

* Dataset inspired by UCI Machine Learning Repository and clinical datasets.
* Streamlit is used to enable the rapid deployment of ML apps.
* TensorFlow/Keras for building the predictive model.

---

## 📧 Contact 
For queries or collaborations, feel free to connect:  
<p align="center">
  <a href="https://www.linkedin.com/in/aritramukherjeeofficial/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://x.com/AritraMofficial" target="_blank">
    <img src="https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
  </a>
  <a href="https://www.instagram.com/aritramukherjee_official/?__pwa=1" target="_blank">
    <img src="https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram">
  </a>
  <a href="https://leetcode.com/u/aritram_official/" target="_blank">
    <img src="https://img.shields.io/badge/LeetCode-%23FFA116.svg?style=for-the-badge&logo=leetcode&logoColor=white" alt="LeetCode">
  </a>
  <a href="https://github.com/AritraOfficial" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-%23181717.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://discord.com/channels/@me" target="_blank">
    <img src="https://img.shields.io/badge/Discord-%237289DA.svg?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="mailto:aritra.work.official@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/Email-%23D14836.svg?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>

---

**If you find this project useful, please consider giving it a ⭐!**
