# 🤟 Real-Time Sign Language Translator  

### 🧠 About the Project  
The **Real-Time Sign Language Translator** is an AI-powered application that recognizes **American Sign Language (ASL)** hand gestures (A–Z, space, delete, and nothing) and translates them into readable text.  
Built using **Python, TensorFlow, Keras, and OpenCV**, this project bridges the communication gap between the hearing and speech-impaired communities through real-time gesture recognition.

---

### 🚀 Features  
- 🖐️ Detects ASL alphabets in real time using a webcam  
- 🧠 Trained a CNN model achieving **99.3% accuracy**  
- 📷 Supports both **live camera feed** and **image upload**  
- 🗨️ Displays predicted letters and forms full words dynamically  
- 🌐 Frontend built using **Streamlit** for a simple, interactive web interface  
- 💾 Model saved in both `.keras` and `.h5` formats for compatibility  

---

### 🧩 Tech Stack  
- **Language:** Python  
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Pillow, Streamlit  
- **Model:** Convolutional Neural Network (CNN)  
- **Dataset:** ASL Alphabet Dataset (A–Z, space, del, nothing)  

---

### ⚙️ Installation & Setup  

#### 1. Clone the Repository  
```bash
git clone https://github.com/yourusername/sign-language-translator.git
cd sign-language-translator


2.  Create a Virtual Environment
    python -m venv venv
    venv\Scripts\activate


3. Install Dependencies
    pip install -r requirements.txt

4. Train the Model (Optional – already provided)    
    python src/train_cnn.py

5. Run the Real-Time Translator
    python src/real_time.py

6. Launch the Web Frontend
    streamlit run app.py

📊 Model Performance
Metric	Value
Training Accuracy	97.08%
Validation Accuracy	99.30%
Validation Loss	0.0234


🖥️ Folder Structure
sign_language_translator/
│
├── dataset/
├── data_split/
├── models/
│   ├── asl_cnn.h5
│   ├── asl_cnn.keras
│
├── src/
│   ├── train_cnn.py
│   ├── prepare_data.py
│   ├── real_time.py
│
├── app.py
├── requirements.txt
└── README.md



🎯 Future Enhancements

🔊 Add text-to-speech output for live translation
📱 Build a mobile-friendly version using Flutter or React Native
🌍 Extend support for dynamic sign gestures and sentences


💡 Acknowledgment

Dataset sourced from Kaggle - ASL Alphabet Dataset
Developed as a Final Year AI/ML Project to promote inclusivity and communication accessibility."# Sign_Language_Translator" 
