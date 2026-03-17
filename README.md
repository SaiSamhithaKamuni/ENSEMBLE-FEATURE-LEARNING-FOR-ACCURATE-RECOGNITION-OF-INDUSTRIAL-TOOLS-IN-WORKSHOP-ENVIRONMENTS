# ENSEMBLE-FEATURE-LEARNING-FOR-ACCURATE-RECOGNITION-OF-INDUSTRIAL-TOOLS-IN-WORKSHOP-ENVIRONMENTS
📌 Project Overview
------------------------------------------------------------------------------------------------------------------------------------------------------------------
This project presents an intelligent Industrial Tool Recognition System that automatically identifies and classifies tools from images using deep learning and machine learning techniques. The system leverages CNN-based feature extraction and ensemble learning models to achieve high accuracy in recognizing tools such as hammer, pliers, gasoline can, and pebble. It aims to reduce manual effort, minimize errors, and support automation in industrial environments.

🚀 Technologies Used
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Python

TensorFlow / Keras

OpenCV

Scikit-learn

NumPy / Pandas

Matplotlib / Seaborn

HTML / CSS (for UI if used)

🔐 Key Features
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Automated industrial tool recognition

Deep learning-based feature extraction (InceptionResNetV2)

Multiple classifiers (Decision Tree, KNN, Perceptron)

Hybrid model (DNN + Random Forest) for high accuracy

Multi-class tool classification

Performance evaluation using Accuracy, ROC, Confusion Matrix

Prediction on real-time/unseen images

⚙️ System Modules
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Dataset Upload Module

Image Preprocessing Module

Feature Extraction Module

Model Training Module

Evaluation Module

Prediction Module

▶️ How to Run Project
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Clone the repository

git clone <your-repo-link>


Install dependencies

pip install -r requirements.txt


Run the project

python main.py

📂 File Arrangement Flowchart
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Project Folder
│
├── dataset/
│   ├── Hammer/
│   ├── Pliers/
│   ├── Gasoline_Can/
│   └── Pebble/
│
├── models/
│   ├── feature_extractor.h5
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── perceptron.pkl
│   └── hybrid_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── training.py
│   ├── evaluation.py
│   └── prediction.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│
├── main.py
├── requirements.txt
└── README.md

🎯 Future Enhancements
------------------------------------------------------------------------------------------------------------------------------------------------------------------

Real-time tool detection using camera

Integration with Industrial IoT systems

Mobile/Web application deployment

Support for more tool categories

👩‍💻 Author
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sai Samhitha Kamuni
