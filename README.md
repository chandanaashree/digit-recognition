 1) Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.  
The model is deployed as an interactive web application using Streamlit.

---
2) Features
- Upload handwritten digit images
- Real-time digit prediction
- Confidence score for predictions
- Visualization of preprocessed input (28×28)
- Clean and user-friendly UI

---
 3) Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Streamlit
- MNIST Dataset

---
 4) Model Architecture
- Convolutional Layers
- MaxPooling Layers
- Fully Connected Dense Layers
- Softmax Output Layer

Trained on the MNIST dataset with validation accuracy above 98%.

---
 5) Project Structure
 
 digit-recognition/
│
├── app.py # Streamlit app
├── cnn_model.keras(Trained CNN model)
├── notebooks/
│ └── mnist_exploration.ipynb
├── requirements.txt
└── README.md
